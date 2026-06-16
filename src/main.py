import asyncio
import json
import re
import warnings
from datetime import datetime
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
    await cleanup_checkpointer()
    print("Server shutting down")


app = FastAPI(title="Design Challenger", lifespan=lifespan)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s]+", "_", text)
    return text.strip("_")[:80]


def _ascii_safe(text: str, fallback: str = "download") -> str:
    """Strip non-ASCII characters for HTTP header safety."""
    ascii_only = text.encode("ascii", errors="ignore").decode("ascii").strip()
    return ascii_only[:30].replace(" ", "_") if ascii_only else fallback


def _save_design_doc(session_id: str):
    session = db.get_session(session_id)
    if not session or not session.get("design_doc"):
        return
    title = session.get("title", "design")
    safe_title = _sanitize_filename(title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_title}_{timestamp}.md"
    filepath = OUTPUT_DIR / filename
    filepath.write_text(session["design_doc"], encoding="utf-8")
    print(f"[Output] Design doc saved to: {filepath}")


def _derive_phase(conversation: list) -> str:
    if not conversation:
        return "Initializing..."
    last = conversation[-1]
    role = last.get("role", "")
    rtype = last.get("type", "")
    rnd = last.get("round", 0)
    if role == "designer" and rtype == "initial":
        return "DesignerExpert generated initial draft"
    elif role == "challenger" and rtype in ("challenge", "human_challenge"):
        label = "Human" if rtype == "human_challenge" else "Challenger"
        return f"{label} raised challenges (round {rnd})"
    elif role == "designer" and rtype == "response":
        return f"DesignerExpert addressed challenges (round {rnd})"
    return f"Round {rnd} ({rtype})"


MODEL_NAME = load_config().get("llm", {}).get("model", "N/A")


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


def _save_state(session_id: str, event: dict, status: str = None):
    updates = dict(
        conversation=_serialize_conversation(event.get("conversation", [])),
        design_doc=event.get("design_doc", ""),
        token_usage=event.get("token_usage", {}).get("total_tokens", 0),
        total_rounds=event.get("current_round", 0),
    )
    if status:
        updates["status"] = status
    db.update_session(session_id, **updates)


@app.get("/api/sessions/{session_id}/stream")
async def stream_session(session_id: str, resume: bool = Query(False)):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    graph = await get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = None if resume else _build_initial_state(session)

    async def event_generator():
        prev_conversation_len = 0
        final_event = None

        # Send model info and initial status
        yield f"data: {json.dumps({'type': 'model_info', 'model': MODEL_NAME}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'phase': 'DesignerExpert is analyzing the requirement and drafting the initial design...', 'token_usage': 0}, ensure_ascii=False)}\n\n"

        try:
            async for event in graph.astream(initial_state, config, stream_mode="values"):
                final_event = event

                if active_sessions.get(session_id, {}).get("stop", False):
                    _save_state(session_id, event, "interrupted")
                    yield f"data: {json.dumps({'type': 'interrupted', 'session_id': session_id}, ensure_ascii=False)}\n\n"
                    return

                conversation = event.get("conversation", [])
                new_msgs = conversation[prev_conversation_len:]
                prev_conversation_len = len(conversation)

                for msg in new_msgs:
                    msg_type = msg.get("type", "")
                    yield f"data: {json.dumps({'type': 'message', 'role': msg['role'], 'content': msg['content'], 'round': msg['round'], 'msg_type': msg_type}, ensure_ascii=False)}\n\n"

                token_total = event.get("token_usage", {}).get("total_tokens", 0)
                phase = _derive_phase(conversation)
                yield f"data: {json.dumps({'type': 'status', 'phase': phase, 'token_usage': token_total}, ensure_ascii=False)}\n\n"

                _save_state(session_id, event)

                if event.get("human_review_pending"):
                    _save_state(session_id, event, "awaiting_human")
                    active_sessions[session_id] = {"awaiting_human": True}
                    yield f"data: {json.dumps({'type': 'human_review_needed', 'session_id': session_id, 'message': 'Challenger has signed off. Do you have any additional challenges?'}, ensure_ascii=False)}\n\n"
                    return

                if event.get("converged"):
                    _save_state(session_id, event, "completed")
                    _save_design_doc(session_id)
                    yield f"data: {json.dumps({'type': 'completed', 'session_id': session_id}, ensure_ascii=False)}\n\n"
                    return

            if final_event:
                _save_state(session_id, final_event, "completed")
                _save_design_doc(session_id)
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
        _save_design_doc(session_id)
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

            yield f"data: {json.dumps({'type': 'status', 'phase': 'Addressing human challenges...', 'token_usage': session.get('token_usage', 0)}, ensure_ascii=False)}\n\n"

            new_messages = await run_human_challenge_loop(state_copy)

            for msg in new_messages:
                yield f"data: {json.dumps({'type': 'message', **msg}, ensure_ascii=False)}\n\n"

            token_total = state_copy.get("token_usage", {}).get("total_tokens", 0)
            phase = _derive_phase(state_copy.get("conversation", []))
            yield f"data: {json.dumps({'type': 'status', 'phase': phase, 'token_usage': token_total}, ensure_ascii=False)}\n\n"

            _save_state(session_id, {
                "conversation": state_copy.get("conversation", []),
                "design_doc": state_copy.get("design_doc", ""),
                "token_usage": state_copy.get("token_usage", {}),
                "current_round": state_copy.get("current_round", 0),
            })

            if state_copy.get("converged"):
                db.update_session(session_id, status="awaiting_human")
                active_sessions[session_id] = {"awaiting_human": True}
                yield f"data: {json.dumps({'type': 'human_review_needed', 'session_id': session_id, 'message': 'Challenger signed off again. Any more challenges?'}, ensure_ascii=False)}\n\n"
            else:
                db.update_session(session_id, status="completed")
                _save_design_doc(session_id)
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


def _condense_log_content(msg: dict) -> str:
    """Condense message content for the challenge log — skip full design doc bodies."""
    content = msg.get("content", "")
    rtype = msg.get("type", "")

    if rtype == "initial":
        # Keep only the first paragraph before any markdown heading
        first = content.split("##")[0].strip()
        return first + "\n\n*[Full initial design document available via Download Design Doc]*"

    if rtype == "response" and "## Updated Design Document" in content:
        # Keep the response-to-challenges part, cut the full updated doc
        parts = content.split("## Updated Design Document", 1)
        return parts[0].strip() + "\n\n*[Updated design document available via Download Design Doc]*"

    return content


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
        condensed = _condense_log_content(msg)
        log_parts.append(f"## {title}\n\n{condensed}\n\n---\n\n")

    log_text = "".join(log_parts)
    safe_title = _ascii_safe(session.get("title", ""), "log")

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
    safe_title = _ascii_safe(session.get("title", ""), "design")

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
