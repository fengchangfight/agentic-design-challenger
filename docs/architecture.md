# Architecture

## Overview

Design Challenger is an adversarial AI document generation system. Two specialized agents (DesignerExpert and Challenger) iterate through a challenge-response loop, with optional human review, to produce high-quality software design documents.

## System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         Web Browser                               │
│  ┌──────────┐  ┌────────────────┐  ┌──────────────────────────┐  │
│  │ Session  │  │  Chat Display   │  │  Human Review Bar        │  │
│  │ Sidebar  │  │  (SSE live)     │  │  (post AI sign-off)      │  │
│  └──────────┘  └────────────────┘  └──────────────────────────┘  │
└──────────────────────┬───────────────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼───────────────────────────────────────────┐
│                     FastAPI (src/main.py)                          │
│                                                                    │
│  REST: create/list/delete sessions, stop, download, human review  │
│  SSE:  stream agent conversation, stream human review loop        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│               LangGraph StateGraph (src/graph.py)                  │
│                                                                    │
│  ┌───────────────────────────┐                                     │
│  │     designer_generate     │  Generates initial design doc       │
│  │  (or respond, on resume)  │                                     │
│  └─────────────┬─────────────┘                                     │
│                ▼                                                    │
│  ┌───────────────────────────┐     ┌─ converged? ──YES──→ END     │
│  │     challenger_review      │─────┤                              │
│  │  Critiques the design doc  │     └─ NO ──→ designer_respond     │
│  └─────────────┬─────────────┘                                     │
│                │                                                    │
│                │  (after sign-off)                                  │
│                ▼                                                    │
│  ┌───────────────────────────┐                                     │
│  │     designer_respond       │  Addresses challenges, updates doc  │
│  │                           │                                     │
│  └─────────────┬─────────────┘                                     │
│                │                                                    │
│                └────→ back to challenger_review                     │
│                                                                    │
│  Checkpoint: SqliteSaver → data/checkpoints.db                    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│                     External Services                              │
│  ┌──────────────────┐   ┌───────────────────────────┐             │
│  │  LLM API          │   │  RAG Vector Store          │             │
│  │  (DeepSeek/etc.)  │   │  (Milvus / MilvusLite)     │             │
│  └──────────────────┘   └───────────────────────────┘             │
│                                                                    │
│  ┌──────────────────┐                                              │
│  │  SQLite            │  data/design_challenger.db                │
│  │  (Session storage) │                                              │
│  └──────────────────┘                                              │
└──────────────────────────────────────────────────────────────────┘
```

## Human-in-the-Loop Flow

```
AI Loop:
  DesignerExpert → Challenger → DesignerExpert → Challenger → ...
                                                    │
                                            signs off?
                                                    │
  ┌─────────────────────────────────────────────────┘
  ▼
  System pauses, asks human: "Any more challenges?"
        │
  ┌─────┴─────┐
  │           │
  Human       Human
  challenges  signs off
  │           │
  ▼           ▼
  Back to     Session
  AI loop     completed
```

When the human provides challenges, the system runs a new **designer_respond → challenger_review** loop to address them. If Challenger signs off again, the human is asked once more. This continues until the human chooses to sign off.

## Data Flow

```
1. User submits requirement + challenge level
2. Backend creates session in SQLite, returns session_id
3. Frontend opens SSE connection: GET /api/sessions/{id}/stream
4. Backend builds initial AgentState, runs graph.astream()
5. Each node completion → state update → SSE event to frontend
6. Frontend appends message to chat display
7. When human_review_needed event arrives:
   a. Frontend shows review bar, enables input
   b. Human types challenge or clicks "Sign Off"
   c. POST /api/sessions/{id}/human_review
   d. If challenge: GET /api/sessions/{id}/stream_human
      Backend runs run_human_challenge_loop(), streams results
      If Challenger signs off → another human_review_needed event
   e. If sign-off: session marked completed
8. Completed session: download buttons enabled, input frozen
```

## State Management

- **LangGraph State**: Managed by `StateGraph(AgentState)` with TypedDict state. The `conversation` list uses an `operator.add` reducer (append-only). All other keys use the default "replace" reducer.
- **Checkpoint**: `SqliteSaver` stores full graph state to `data/checkpoints.db` after each node. Enables resume on server restart.
- **Application DB**: `data/design_challenger.db` stores session metadata, conversation (as JSON), design_doc, token_usage, etc. This is the primary data store; the checkpoint is for runtime state only.

## Resilience

- **Stop mid-run**: User clicks Stop → `active_sessions[id].stop = True` → SSE loop detects flag after current node → saves state → marks "interrupted"
- **Page close**: SSE connection drops → `asyncio.CancelledError` caught → state saved → "interrupted"
- **Server crash**: Checkpoint in `SqliteSaver` survives. On restart, session shows "Continue" button → resumes from last checkpoint.
- **LLM errors**: Caught as exceptions in SSE loop → emitted as `error` events → session marked "interrupted"

## Extensibility

- **New LLM provider**: Edit `config/llm.yaml` — any OpenAI-compatible API works
- **New challenge levels**: Add entries to `CHALLENGER_PROMPTS` dict in `agents.py`
- **RAG ingestion**: Extend `src/rag.py` with document loading/indexing functions
- **Multiple reviewers**: Add more agent nodes to the graph (e.g., SecurityReviewer, PerformanceReviewer)
- **Tool calling**: Bind tools to the LLM in agents.py instead of prompt-injection for RAG
