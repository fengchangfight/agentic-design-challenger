import yaml
from pathlib import Path
from typing import List, Optional

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

CONFIG_PATH = Path(__file__).parent.parent / "config" / "rag.yaml"
DATA_DIR = Path(__file__).parent.parent / "data"

_index: Optional[VectorStoreIndex] = None


def _load_rag_config():
    with open(str(CONFIG_PATH), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["rag"]


def _get_embedding_model(rag_config: dict):
    return OpenAIEmbedding(
        api_base=rag_config["embedding_base_url"],
        api_key=rag_config["embedding_api_key"],
        model=rag_config["embedding_model"],
    )


def get_index() -> Optional[VectorStoreIndex]:
    global _index
    if _index is not None:
        return _index

    rag_config = _load_rag_config()
    embed_model = _get_embedding_model(rag_config)
    Settings.embed_model = embed_model

    mode = rag_config.get("mode", "milvus_lite")

    try:
        if mode == "milvus_lite":
            db_path = DATA_DIR / rag_config.get("milvus_lite_db", "milvus_lite.db")
            db_path.parent.mkdir(exist_ok=True)
            vector_store = MilvusVectorStore(
                uri=str(db_path),
                collection_name=rag_config.get("collection_name", "design_knowledge"),
                dim=1536,
                overwrite=False,
            )
        else:
            host = rag_config.get("milvus_host", "127.0.0.1")
            port = rag_config.get("milvus_port", 19530)
            vector_store = MilvusVectorStore(
                uri=f"http://{host}:{port}",
                collection_name=rag_config.get("collection_name", "design_knowledge"),
                dim=1536,
                overwrite=False,
            )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _index = VectorStoreIndex([], storage_context=storage_context)
        return _index
    except Exception as e:
        print(f"[RAG] Warning: Could not connect to knowledge base: {e}")
        return None


def search_knowledge(query: str, top_k: int = None) -> List[str]:
    """Search the knowledge base for relevant information."""
    rag_config = _load_rag_config()
    if top_k is None:
        top_k = rag_config.get("top_k", 5)

    index = get_index()
    if index is None:
        return []

    try:
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        results = []
        for node in nodes:
            text = node.get_content()
            if text and text.strip():
                results.append(text.strip())
        return results
    except Exception as e:
        print(f"[RAG] Search error: {e}")
        return []


def format_knowledge_context(results: List[str]) -> str:
    """Format search results for inclusion in LLM context."""
    if not results:
        return ""
    parts = ["## Relevant Knowledge from Knowledge Base\n"]
    for i, result in enumerate(results, 1):
        parts.append(f"### Reference {i}\n{result}\n")
    return "\n".join(parts)
