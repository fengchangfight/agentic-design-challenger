import asyncio
import json
import warnings
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import fastapi
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from pydantic import BaseModel

warnings.filterwarnings("ignore", message=".*PyTorch.*")
warnings.filterwarnings("ignore", message=".*TensorFlow.*")

from src.config import load_config
from src.db import SessionDB
from src.graph import get_compiled_graph, cleanup_checkpointer, run_human_challenge_loop

db = SessionDB()

active_sessions: dict = {}
"""session_id -> {"stop": bool, "awaiting_human": bool}"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    host = config["server"]["host"]
    port = config["server"]["port"]
    print(f"Design Challenger server starting on http://{host}:{port}")
    yield
    cleanup_checkpointer()
    print("Server shutting down")


app = FastAPI(title="Design Challenger", lifespan=lifespan)


def _extract_title(text: str) -> str:
    if not text or not text.strip():
        return "Untitled"
    first_line = text.strip().split("\n")[0]
    if first_line.startswith("#"):
        first_line = first_line.lstrip("#").strip()
    return first_line[:100] if first_line else "Untitled"


def _serialize_conversation(conv) -> str:
    if isinstance(conv, str):
        return conv
    return json.dumps(conv, ensure_ascii=False)


def _build_initial_state(session: dict) -> dict:
    return {
        "requirement": session["requirement"],
        "challenge_level": session["challenge_level"],
        "current_round": 0,
        "design_doc": "",
        "conversation": [],
        "converged": False,
        "human_review_pending": False,
        "skip_initial_generation": False,
        "status": "running",
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


@app.post("/api/sessions")
async def create_session(
    requirement: str = Form(""),
    challenge_level: str = Form("medium"),
    file: Optional[UploadFile] = File(None),
):
    content = requirement
    if file and file.filename:
        file_content = await file.read()
        content = file_content.decode("utf-8")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Requirement is required")
    if challenge_level not in ("weak", "medium", "strong"):
        raise HTTPException(status_code=400, detail="Invalid challenge level")

    title = _extract_title(content)
    session_id = db.create_session(content, challenge_level, title)
    return {"session_id": session_id, "status": "created"}


@app.get("/api/sessions/{session_id}/stream")
async def stream_session(session_id: str, resume: bool = Query(False)):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = None if resume else _build_initial_state(session)

    async def event_generator():
        prev_conversation_len = 0
        final_event = None

        try:
            async for event in graph.astream(initial_state, config, stream_mode="values"):
                final_event = event

                if active_sessions.get(session_id, {}).get("stop", False):
                    db.update_session(
                        session_id,
                        status="interrupted",
                        conversation=_serialize_conversation(event.get("conversation", [])),
                        design_doc=event.get("design_doc", ""),
                        token_usage=event.get("token_usage", {}).get("total_tokens", 0),
                        total_rounds=event.get("current_round", 0),
                    )
                    yield f"data: {json.dumps({'type': 'interrupted', 'session_id': session_id}, ensure_ascii=False)}\n\n"
                    return

                conversation = event.get("conversation", [])
                new_msgs = conversation[prev_conversation_len:]
                prev_conversation_len = len(conversation)

                for msg in new_msgs:
                    yield f"data: {json.dumps({'type': 'message', **msg}, ensure_ascii=False)}\n\n"

                db.update_session(
                    session_id,
                    conversation=_serialize_conversation(conversation),
                    design_doc=event.get("design_doc", ""),
                    token_usage=event.get("token_usage", {}).get("total_tokens", 0),
                    total_rounds=event.get("current_round", 0),
                )

                if event.get("human_review_pending"):
                    db.update_session(
                        session_id,
                        status="awaiting_human",
                        conversation=_serialize_conversation(event.get("conversation", [])),
                        design_doc=event.get("design_doc", ""),
                        token_usage=event.get("token_usage", {}).get("total_tokens", 0),
                        total_rounds=event.get("current_round", 0),
                    )
                    active_sessions[session_id] = {"awaiting_human": True}
                    yield f"data: {json.dumps({'type': 'human_review_needed', 'session_id': session_id, 'message': 'Challenger has signed off. Do you have any additional challenges? Type your challenges or sign off.'}, ensure_ascii=False)}\n\n"
                    return

                if event.get("converged"):
                    db.update_session(
                        session_id,
                        status="completed",
                        conversation=_serialize_conversation(event.get("conversation", [])),
                        design_doc=event.get("design_doc", ""),
                        token_usage=event.get("token_usage", {}).get("total_tokens", 0),
                        total_rounds=event.get("current_round", 0),
                    )
                    yield f"data: {json.dumps({'type': 'completed', 'session_id': session_id}, ensure_ascii=False)}\n\n"
                    return

            if final_event:
                db.update_session(
                    session_id,
                    status="completed",
                    conversation=_serialize_conversation(final_event.get("conversation", [])),
                    design_doc=final_event.get("design_doc", ""),
                    token_usage=final_event.get("token_usage", {}).get("total_tokens", 0),
                    total_rounds=final_event.get("current_round", 0),
                )
                yield f"data: {json.dumps({'type': 'completed', 'session_id': session_id}, ensure_ascii=False)}\n\n"

        except asyncio.CancelledError:
            if final_event:
                db.update_session(
                    session_id,
                    status="interrupted",
                    conversation=_serialize_conversation(final_event.get("conversation", [])),
                    design_doc=final_event.get("design_doc", ""),
                    token_usage=final_event.get("token_usage", {}).get("total_tokens", 0),
                    total_rounds=final_event.get("current_round", 0),
                )
        except Exception as e:
            db.update_session(session_id, status="interrupted")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            active_sessions.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class HumanReviewRequest(BaseModel):
    text: str


@app.post("/api/sessions/{session_id}/human_review")
async def human_review(session_id: str, body: HumanReviewRequest):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    human_text = body.text.strip()
    if not human_text:
        raise HTTPException(status_code=400, detail="Input is required")

    signoff_keywords = ["我sign off", "没有更多疑问", "sign off", "signoff", "sign off"]
    is_signoff = any(kw in human_text.lower() for kw in signoff_keywords)

    if is_signoff:
        db.update_session(
            session_id,
            status="completed",
            conversation=_serialize_conversation(session.get("conversation", [])),
            design_doc=session.get("design_doc", ""),
        )
        return {"status": "completed", "message": "Session completed by human sign-off"}

    # Append human challenge to conversation
    conversation = session.get("conversation", [])
    if isinstance(conversation, str):
        conversation = json.loads(conversation)
    rnd = session.get("total_rounds", 1)
    conversation.append({
        "role": "challenger",
        "content": human_text,
        "round": rnd,
        "type": "human_challenge"
    })
    db.update_session(
        session_id,
        conversation=_serialize_conversation(conversation),
        status="running",
    )

    return {"status": "challenge_accepted", "stream_url": f"/api/sessions/{session_id}/stream_human"}


@app.get("/api/sessions/{session_id}/stream_human")
async def stream_human_review(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    conversation = session.get("conversation", [])
    if isinstance(conversation, str):
        conversation = json.loads(conversation)

    # The last message should be the human's challenge (appended by frontend)
    # Build state from DB
    state = {
        "requirement": session["requirement"],
        "challenge_level": session.get("challenge_level", "medium"),
        "current_round": session.get("total_rounds", 1),
        "design_doc": session.get("design_doc", ""),
        "conversation": conversation.copy(),
        "converged": False,
        "human_review_pending": False,
        "skip_initial_generation": True,
        "status": "running",
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": session.get("token_usage", 0),
        },
    }

    async def event_generator():
        try:
            state_copy = dict(state)
            state_copy["conversation"] = list(state.get("conversation", []))

            new_messages = await run_human_challenge_loop(state_copy)

            for msg in new_messages:
                yield f"data: {json.dumps({'type': 'message', **msg}, ensure_ascii=False)}\n\n"

            db.update_session(
                session_id,
                conversation=_serialize_conversation(state_copy.get("conversation", [])),
                design_doc=state_copy.get("design_doc", ""),
                token_usage=state_copy.get("token_usage", {}).get("total_tokens", 0),
                total_rounds=state_copy.get("current_round", 0),
            )

            if state_copy.get("converged"):
                db.update_session(
                    session_id,
                    status="awaiting_human",
                )
                active_sessions[session_id] = {"awaiting_human": True}
                yield f"data: {json.dumps({'type': 'human_review_needed', 'session_id': session_id, 'message': 'Challenger has signed off again. Any more challenges?'}, ensure_ascii=False)}\n\n"
            else:
                db.update_session(
                    session_id,
                    status="completed",
                )
                yield f"data: {json.dumps({'type': 'completed', 'session_id': session_id}, ensure_ascii=False)}\n\n"

        except Exception as e:
            db.update_session(session_id, status="interrupted")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            active_sessions.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    active_sessions[session_id] = {"stop": True}
    return {"status": "stopping"}


@app.get("/api/sessions")
async def list_sessions():
    sessions = db.list_sessions()
    return {"sessions": sessions}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    db.delete_session(session_id)
    return {"status": "deleted"}


@app.get("/api/sessions/{session_id}/download/log")
async def download_log(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    conversation = session.get("conversation", [])
    if isinstance(conversation, str):
        conversation = json.loads(conversation)

    log_parts = [
        f"# Design Challenge Log\n",
        f"**Topic**: {session.get('title', 'Untitled')}\n",
        f"**Challenge Level**: {session.get('challenge_level', 'N/A')}\n",
        f"**Total Rounds**: {session.get('total_rounds', 0)}\n",
        f"**Token Usage**: {session.get('token_usage', 0)}\n",
        f"**Created**: {session.get('created_at', 'N/A')}\n",
        f"**Status**: {session.get('status', 'N/A')}\n\n",
        f"---\n\n",
    ]

    for msg in conversation:
        role = msg.get("role", "unknown").upper()
        rtype = msg.get("type", "")
        rnd = msg.get("round", "?")
        title = f"{role} (Round {rnd}, {rtype})"
        log_parts.append(f"## {title}\n\n{msg.get('content', '')}\n\n---\n\n")

    log_text = "".join(log_parts)
    safe_title = session.get("title", "log")[:30].replace(" ", "_").replace("/", "_")

    return Response(
        content=log_text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=challenge_log_{session_id[:8]}_{safe_title}.md"},
    )


@app.get("/api/sessions/{session_id}/download/doc")
async def download_doc(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    design_doc = session.get("design_doc", "")
    safe_title = session.get("title", "design")[:30].replace(" ", "_").replace("/", "_")

    return Response(
        content=design_doc,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=design_doc_{session_id[:8]}_{safe_title}.md"},
    )


STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Design Challenger</h1><p>index.html not found</p>")
