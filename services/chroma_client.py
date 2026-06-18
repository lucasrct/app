"""ChromaDB client management with singleton pattern."""

import os
import functools
from typing import Optional, Callable, Any, Dict

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config import get_config

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "text-embedding-3-small"


class ChromaClientManager:
    """Singleton manager for ChromaDB PersistentClient connections.

    Each collection stores its embedding provider and model name in its
    ChromaDB metadata so the right embedding function is always used,
    regardless of which process or session opens the collection later.
    """

    _instance: Optional["ChromaClientManager"] = None
    _client: Optional[chromadb.ClientAPI] = None
    _ef_cache: Dict[str, Any] = {}

    def __new__(cls) -> "ChromaClientManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            config = get_config()
            self._config = config
            self._client = chromadb.PersistentClient(
                path=config.chroma.persist_directory
            )
            ChromaClientManager._ef_cache = {}

    # ── Embedding functions ────────────────────────────────────────────────

    def get_embedding_function(self, provider: str, model_name: str):
        """Return a cached embedding function for the given provider/model."""
        key = f"{provider}:{model_name}"
        if key not in self._ef_cache:
            if provider == "sentence_transformers":
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                self._ef_cache[key] = SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
            else:  # openai
                api_key = self._config.openai_api_key or os.getenv("OPENAI_API_KEY")
                self._ef_cache[key] = OpenAIEmbeddingFunction(
                    model_name=model_name,
                    api_key=api_key,
                    api_key_env_var="OPENAI_API_KEY",
                )
        return self._ef_cache[key]

    @property
    def client(self) -> chromadb.ClientAPI:
        return self._client

    # ── Collection access ──────────────────────────────────────────────────

    def get_collection(
        self,
        name: str,
        provider: str = DEFAULT_PROVIDER,
        model_name: str = DEFAULT_MODEL,
    ) -> chromadb.Collection:
        """Get or create a collection, persisting its embedding spec in metadata."""
        ef = self.get_embedding_function(provider, model_name)
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=ef,
            metadata={
                "embedding_provider": provider,
                "embedding_model": model_name,
            },
        )

    def get_existing_collection(self, name: str) -> Optional[chromadb.Collection]:
        """Get an existing collection using the embedding function from its metadata."""
        try:
            # First call — no EF needed, just to read stored metadata
            raw = self._client.get_collection(name)
            meta = raw.metadata or {}
            provider = meta.get("embedding_provider", DEFAULT_PROVIDER)
            model_name = meta.get("embedding_model", DEFAULT_MODEL)
            ef = self.get_embedding_function(provider, model_name)
            # Second call — with the right EF attached
            return self._client.get_collection(name, embedding_function=ef)
        except Exception:
            return None

    def get_collection_embedding_info(self, name: str) -> Dict[str, str]:
        """Return the embedding provider/model stored in a collection's metadata."""
        try:
            raw = self._client.get_collection(name)
            meta = raw.metadata or {}
            return {
                "embedding_provider": meta.get("embedding_provider", DEFAULT_PROVIDER),
                "embedding_model": meta.get("embedding_model", DEFAULT_MODEL),
            }
        except Exception:
            return {
                "embedding_provider": DEFAULT_PROVIDER,
                "embedding_model": DEFAULT_MODEL,
            }

    def list_collections(self) -> list:
        return self._client.list_collections()

    def delete_collection(self, name: str) -> bool:
        try:
            self._client.delete_collection(name)
            return True
        except Exception:
            return False

    def heartbeat(self) -> int:
        return self._client.heartbeat()

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls._client = None
        cls._ef_cache = {}


def get_chroma_client() -> ChromaClientManager:
    return ChromaClientManager()


def require_collection(f: Callable) -> Callable:
    """Decorator: injects a resolved ChromaDB collection into the wrapped function."""
    @functools.wraps(f)
    def wrapper(*args, collection_name: str = "", **kwargs) -> Any:
        if not collection_name:
            raise ValueError("collection_name is required")
        manager = get_chroma_client()
        collection = manager.get_existing_collection(collection_name)
        if collection is None:
            raise ValueError(f"Collection not found: {collection_name}")
        kwargs["collection"] = collection
        return f(*args, **kwargs)
    return wrapper
