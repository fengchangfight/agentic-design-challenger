# Design Challenger

Adversarial AI system: two LangGraph agents (DesignerExpert + Challenger) debate to produce high-quality design documents, with human-in-the-loop review.

## Stack

- **Python 3.10+**, **LangGraph 1.x** (StateGraph + SqliteSaver checkpoint)
- **LangChain + langchain-openai** (LLM calls via OpenAI-compatible API, e.g. DeepSeek)
- **FastAPI + uvicorn** (async web server, SSE streaming)
- **SQLite** (session storage + LangGraph checkpoint)
- **LlamaIndex + Milvus** (RAG knowledge retrieval, milvus_lite local mode default)
- **Vanilla HTML/CSS/JS** (no frontend framework)

## Directory Layout

```
design-challenger/
├── agents.md                  # This file (AI agent instructions)
├── run.py                     # Entry point: suppresses warnings, starts uvicorn
├── requirements.txt
├── .gitignore
├── config/
│   ├── llm.yaml               # LLM: base_url, api_key, model, temperature
│   └── rag.yaml               # RAG: mode (milvus_lite/milvus), embedding, top_k
├── data/                      # Auto-created: *.db files (gitignored)
├── docs/                      # Human-facing docs
│   ├── architecture.md
│   └── usage.md
└── src/
    ├── __init__.py
    ├── config.py              # YAML config loader (caches in memory)
    ├── agents.py              # System prompts for DesignerExpert + Challenger, LLM factory
    ├── rag.py                 # LlamaIndex + Milvus vector search, format_knowledge_context()
    ├── db.py                  # SQLite CRUD for sessions table (SessionDB class)
    ├── graph.py               # LangGraph StateGraph: 3 nodes, conditional edges, SqliteSaver
    ├── main.py                # FastAPI app: REST endpoints + SSE streaming
    └── static/
        └── index.html         # Single-page web UI
```

## Key Files

### src/graph.py

`AgentState` (TypedDict) fields:
- `requirement: str` — user requirement text
- `challenge_level: str` — "weak"|"medium"|"strong"
- `current_round: int` — monotonically increasing round counter
- `design_doc: str` — latest complete design document
- `conversation: Annotated[List[dict], operator.add]` — append-only message history, each dict has `role, content, round, type`
- `converged: bool` — Challenger said "我sign off，没有更多疑问"
- `human_review_pending: bool` — human needs to review before final completion
- `skip_initial_generation: bool` — when True, designer_generate_node delegates to designer_respond_node
- `token_usage: Dict[str,int]` — accumulated {input_tokens, output_tokens, total_tokens}

Three async nodes:
1. **designer_generate_node** — generates initial design doc. If `skip_initial_generation`, acts as designer_respond.
2. **challenger_review_node** — reviews design, raises challenges. Checks for `CONVERGENCE_MARKER` in output. Sets `human_review_pending=True` on convergence.
3. **designer_respond_node** — addresses each challenge point-by-point, outputs full updated design doc.

Graph topology: `designed_generate → challenger_review → (should_continue?) → designer_respond → challenger_review ...`

`should_continue` returns "end" if converged or `current_round > safety_max_rounds` (default 20).

`run_human_challenge_loop(state)` — standalone loop called after human submits challenges; calls designer_respond and challenger_review nodes directly until convergence, then returns.

### src/agents.py

- `get_llm()` — singleton `ChatOpenAI` from config/llm.yaml
- `DESIGNER_SYSTEM_PROMPT` — long system prompt for DesignerExpert
- `CHALLENGER_PROMPTS` — dict of 3 prompts keyed by level
- `CONVERGENCE_MARKER = "我sign off，没有更多疑问"`

### src/rag.py

- `get_index()` — lazy-init VectorStoreIndex (MilvusVectorStore, supports local `.db` URI or remote HTTP)
- `search_knowledge(query, top_k=5)` — vector similarity search
- `format_knowledge_context(results)` — formats results as markdown for prompt injection
- Search results are injected into LLM prompts in every graph node call.

### src/db.py

`SessionDB` class wrapping sqlite3:
- Table: `sessions(id TEXT PK, title, requirement, challenge_level, status, design_doc, conversation TEXT/JSON, token_usage INT, total_rounds INT, created_at, updated_at)`
- Statuses: `pending`, `running`, `completed`, `interrupted`, `awaiting_human`

### src/main.py

FastAPI app with lifespan (inits/cleans checkpointer). Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/sessions` | Create session (Form: requirement+level, File: .md upload) |
| GET | `/api/sessions/{id}/stream?resume=` | SSE stream of agent conversation |
| POST | `/api/sessions/{id}/stop` | Set stop flag on running session |
| GET | `/api/sessions` | List all sessions (JSON) |
| DELETE | `/api/sessions/{id}` | Delete session + data |
| POST | `/api/sessions/{id}/human_review` | Submit human challenge or sign-off |
| GET | `/api/sessions/{id}/stream_human` | SSE stream after human challenge |
| GET | `/api/sessions/{id}/download/log` | Download conversation as .md |
| GET | `/api/sessions/{id}/download/doc` | Download design doc as .md |

SSE events: `message`, `human_review_needed`, `completed`, `interrupted`, `error`.

### src/static/index.html

Single-page vanilla JS app. Sections:
- Left sidebar: session list (auto-refresh every 5s)
- Input area: textarea, file upload, level selector
- Chat area: real-time message rendering via SSE
- Human review bar: appears when AI signs off, has input + "Submit Challenge" / "Sign Off & Finish" buttons
- Action bar: download buttons, continue button (for interrupted sessions)

## Graph Flow

```
User starts challenge
  │
  ▼
designer_generate ──→ challenger_review ──→ converged? ──YES──→ human_review_needed event
                           │                                        │
                           │ NO                              ┌──────┘
                           ▼                                 │
                     designer_respond ◄──────────────────────┘
                           │                                 
                           └──→ challenger_review (loop)     
                                                                  
Human review:
  If human challenges → POST human_review → designer responds → challenger reviews → sign off? → ask human again
  If human signs off  → session completed
```

## How to Run

```bash
pip install -r requirements.txt
# Edit config/llm.yaml with your API key
python run.py
# Open http://127.0.0.1:8000
```

## How to Test

No automated tests yet. Manual verification:
1. Start server, open browser
2. Enter a short requirement (e.g. "Design a URL shortener service")
3. Select "weak" level for fast iteration
4. Click Start Challenge, verify SSE streaming
5. Click Stop mid-run, verify session shows "Continue"
6. Click Continue, verify resume
7. After AI sign-off, enter a human challenge, verify loop
8. Sign off, verify download buttons work
9. Delete session, verify sidebar updates

## Conventions

- LangGraph state keys without `Annotated` use default "replace" reducer
- `conversation` uses `operator.add` reducer (append-only)
- Token usage accumulates across nodes via `_accumulate_tokens()`
- Design doc extraction: splits on `"## Updated Design Document"` marker
- Human input sign-off detection: checks for keywords in lowercase
- All file paths use `pathlib.Path`, all DB paths relative to project root
- RAG initialization is lazy and failure-tolerant (returns None, agents work without it)
