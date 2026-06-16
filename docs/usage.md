# Usage Guide

## Prerequisites

- Python 3.10 or higher
- pip

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure LLM

Edit `config/llm.yaml`:

```yaml
llm:
  base_url: "https://api.deepseek.com/v1"   # Your LLM API endpoint
  api_key: "sk-your-actual-api-key"          # Your API key
  model: "deepseek-chat"                     # Model name
  temperature: 0.7

server:
  host: "127.0.0.1"
  port: 8000

challenge:
  safety_max_rounds: 20                      # Max rounds before forced stop
```

The system uses an OpenAI-compatible API. Works with DeepSeek, OpenAI, or any compatible provider.

### 3. Configure RAG (Optional)

Edit `config/rag.yaml`:

```yaml
rag:
  mode: "milvus_lite"              # "milvus_lite" (local) or "milvus" (remote server)
  milvus_lite_db: "data/milvus_lite.db"
  collection_name: "design_knowledge"
  embedding_base_url: "https://api.deepseek.com/v1"
  embedding_api_key: "your-api-key"
  embedding_model: "text-embedding-3-small"
  top_k: 5
```

Default mode `milvus_lite` requires no Milvus server — just a local file.

### 4. Start the server

```bash
python run.py
```

Open your browser to **http://127.0.0.1:8000**

## Using the Web UI

### Starting a Challenge

1. **Enter your requirement** — Paste a design requirement in the text area.
   ```
   Design a real-time chat application supporting 10k concurrent users.
   It should have message history, user presence, and file sharing.
   ```
2. **Or upload a .md file** — Click "Load File" to upload a markdown file containing your requirement.
3. **Select challenge level** — Weak / Medium / Strong (see table below).
4. **Click "Start Challenge"** — The agents begin their adversarial design process.

### Challenge Levels

| Level | Challenger Behavior | Questions per Round |
|-------|---------------------|---------------------|
| **Weak** | Gentle, constructive feedback | 1–3 |
| **Medium** | Thorough: architecture, security, scalability, performance | 3–6 |
| **Strong** | Extremely critical, attacks every angle | 5–10 |

### During the Challenge

- The **chat display** updates in real-time via SSE.
- Messages are color-coded: green (DesignerExpert), red (Challenger), orange (Human).
- You can click **Stop** at any time to pause the process. The session can be resumed later.

### Human Review

When the Challenger signs off, a review bar appears:

> 👁 Challenger signed off. Your review: `[___________]` [Submit Challenge] [Sign Off & Finish]

- **Submit Challenge**: Type your concerns and click this button. The DesignerExpert will address them, and the Challenger will review again. If the Challenger signs off, you'll be asked for review once more.
- **Sign Off & Finish**: Click when you're satisfied. The session completes and download buttons become available.

### After Completion

- **Download Log**: Downloads the full conversation as a `.md` file.
- **Download Design Doc**: Downloads the final design document as a `.md` file.
- The input area is frozen. To start a new design, type a new requirement and click "Start Challenge".

### Session Management

- **Left sidebar** shows all sessions with their status (running, completed, interrupted, awaiting human).
- **Click a session** to view its conversation and design doc.
- **Delete a session** by clicking the ✕ button (confirmation required).
- **Continue an interrupted session** by selecting it and clicking "Continue".

## Command Line

You can also interact with the API directly:

```bash
# List all sessions
curl http://127.0.0.1:8000/api/sessions

# Create a session
curl -X POST http://127.0.0.1:8000/api/sessions \
  -F "requirement=Design a URL shortener" \
  -F "challenge_level=medium"

# Stream the challenge (SSE)
curl -N http://127.0.0.1:8000/api/sessions/{session_id}/stream

# Submit human review (sign-off)
curl -X POST http://127.0.0.1:8000/api/sessions/{session_id}/human_review \
  -H "Content-Type: application/json" \
  -d '{"text": "我sign off，没有更多疑问"}'

# Download the design document
curl http://127.0.0.1:8000/api/sessions/{session_id}/download/doc \
  -o design.md
```

## Configuration Reference

### llm.yaml

| Key | Description | Default |
|-----|-------------|---------|
| `llm.base_url` | LLM API endpoint | `https://api.deepseek.com/v1` |
| `llm.api_key` | API authentication key | (required) |
| `llm.model` | Model identifier | `deepseek-chat` |
| `llm.temperature` | Response randomness (0–1) | `0.7` |
| `server.host` | Bind address | `127.0.0.1` |
| `server.port` | HTTP port | `8000` |
| `challenge.safety_max_rounds` | Max rounds before forced stop | `20` |

### rag.yaml

| Key | Description | Default |
|-----|-------------|---------|
| `rag.mode` | `milvus_lite` or `milvus` | `milvus_lite` |
| `rag.milvus_lite_db` | Local DB path | `data/milvus_lite.db` |
| `rag.milvus_host` | Remote Milvus host | `127.0.0.1` |
| `rag.milvus_port` | Remote Milvus port | `19530` |
| `rag.collection_name` | Vector collection name | `design_knowledge` |
| `rag.embedding_base_url` | Embedding API endpoint | (same as LLM) |
| `rag.embedding_api_key` | Embedding API key | (same as LLM) |
| `rag.embedding_model` | Embedding model name | `text-embedding-3-small` |
| `rag.top_k` | Results per search | `5` |

## Troubleshooting

**"Module not found" errors**: Run `pip install -r requirements.txt` to install all dependencies.

**LLM connection errors**: Verify `config/llm.yaml` has the correct `base_url` and `api_key`. The system uses an OpenAI-compatible API — ensure your provider supports the `/v1/chat/completions` endpoint.

**RAG connection warnings**: If you don't need RAG, ignore the `[RAG] Warning: Could not connect to knowledge base` message. The system works without it.

**PyTorch/TensorFlow warnings**: These come from the `tiktoken` library dependency and are harmless. The system uses a remote LLM API, not local models. Warnings are suppressed at startup.
