# Configuration

All configuration is centralised in `config.py` as a tree of frozen dataclasses. The root object is `AppConfig`, which contains four sub-configs: `ChromaConfig`, `SearchConfig`, `IngestionConfig`, and fields for the Flask application itself.

## Reading Configuration

Call `get_config()` to obtain the current configuration. The function constructs an `AppConfig` from environment variables on every call, so do not call it in a tight loop:

```python
from config import get_config

config = get_config()
print(config.ingestion.max_tokens_per_chunk)  # 1000
```

Services call `get_config()` in their constructor and store the relevant sub-config as an instance attribute. They do not call `get_config()` at request time.

## AppConfig

| Field | Type | Default | Source |
|-------|------|---------|--------|
| `environment` | `Environment` | `development` | `FLASK_ENV` env var |
| `secret_key` | str | `"dev-secret-key-..."` | `SECRET_KEY` env var |
| `host` | str | `"0.0.0.0"` | hardcoded |
| `port` | int | `5000` | hardcoded |
| `debug` | bool | `True` in development | derived from `environment` |
| `openai_api_key` | str \| None | None | `OPENAI_API_KEY` env var |

The `Environment` enum has three members: `DEVELOPMENT`, `PRODUCTION`, and `TESTING`. When `environment` is `DEVELOPMENT`, `debug` is automatically set to `True`, which enables the Flask reloader and the interactive debugger.

## ChromaConfig

Controls how the app connects to ChromaDB.

| Field | Type | Default | Source |
|-------|------|---------|--------|
| `persist_directory` | str | `"./chroma_data"` | `CHROMA_PERSIST_DIR` env var |
| `default_collection` | str | `"code_collection"` | hardcoded |
| `embedding_model` | `EmbeddingModel` | `OPENAI_SMALL` | hardcoded |
| `batch_size` | int | `100` | hardcoded |
| `max_results` | int | `20` | hardcoded |

`persist_directory` determines where ChromaDB writes its SQLite database and vector index files. The directory is created automatically if it does not exist. Use an absolute path when running the app from different working directories.

`embedding_model` is an enum with three members: `OPENAI_SMALL` (`text-embedding-3-small`), `OPENAI_LARGE` (`text-embedding-3-large`), and `DEFAULT` (ChromaDB's built-in `all-MiniLM-L6-v2`, which requires no API key). Changing the embedding model after a collection has been populated will cause semantic searches to produce incorrect results because the stored vectors will not be comparable to new query vectors. Always re-ingest when changing embedding models.

## SearchConfig

Controls query validation and result limits.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_n_results` | int | `10` | Results returned when caller does not specify |
| `max_n_results` | int | `50` | Hard cap on results per query |
| `min_query_length` | int | `2` | Queries shorter than this are rejected with `400` |
| `max_query_length` | int | `500` | Queries longer than this are rejected with `400` |
| `score_precision` | int | `4` | Decimal places in serialised similarity scores |
| `regex_max_results` | int | `100` | Hard cap on regex search results |
| `regex_timeout_seconds` | float | `5.0` | Maximum wall time for a regex search operation |

The `regex_timeout_seconds` limit is not yet enforced at the ChromaDB query level — it is checked after the results come back. A collection with millions of chunks and a pathological regex pattern could still cause a slow response. This is a known limitation.

## IngestionConfig

Controls the ingestion pipeline behaviour.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens_per_chunk` | int | `1000` | Maximum tokens per chunk (tiktoken count) |
| `supported_extensions` | tuple | `(".py", ".md", ".txt")` | File types to ingest |
| `ignore_patterns` | tuple | `("__pycache__", ".git", ".env", "node_modules")` | Directory names to skip |
| `batch_size` | int | `100` | Chunks uploaded per ChromaDB `add()` call |
| `tokenizer_model` | str | `"text-embedding-3-small"` | Model name for tiktoken encoder lookup |
| `fallback_encoding` | str | `"cl100k_base"` | Tiktoken encoding used if model lookup fails |

`max_tokens_per_chunk` is the primary lever for controlling chunk granularity. Smaller values produce more chunks with less context per chunk; larger values produce fewer chunks but risk exceeding the embedding model's context window. The `text-embedding-3-small` model has an 8,191-token limit, so values up to 4,000 are safe, but values above 1,000 will produce chunks that are difficult to display in the UI.

`ignore_patterns` is matched against directory names only, not file names. To exclude specific file types rather than directories, remove them from `supported_extensions`.

## Extending Configuration

To add a new configuration field, add it to the appropriate dataclass in `config.py`. If the field should be settable via an environment variable, read the variable in `AppConfig.from_environment()` or the relevant sub-config's `__post_init__`:

```python
@dataclass
class IngestionConfig:
    max_tokens_per_chunk: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "1000"))
    )
```

Do not read environment variables at module import time outside of dataclass field defaults — this makes it impossible to override values in tests without patching `os.environ` globally.
