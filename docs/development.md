# Development Guide

## Setting Up a Local Environment

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key with access to `text-embedding-3-small`
- Git

### Installation Steps

```bash
# Clone the repository and enter the app directory
git clone <repo-url>
cd app

# Install Python dependencies
pip install -r requirements.txt

# Set your OpenAI key
export OPENAI_API_KEY=sk-...

# Start the development server
FLASK_ENV=development python app.py
```

Visit `http://localhost:5000` in a browser. The first thing you will want to do is ingest a directory so there is a collection to search. Use the Collections page to create a new collection and point it at `../demo_project` (relative to where you started the server) or at the `app/` directory itself.

### Reloading

Flask's built-in reloader watches `.py` files under the working directory and restarts the server when any of them change. HTML templates are reloaded per-request in development mode without a server restart. Static files (JavaScript, CSS) are served directly from disk and are picked up immediately on browser refresh.

## Project Structure Walkthrough

### `app.py` — Application Factory

`create_app()` is the Flask application factory. It applies the `ReverseProxied` WSGI middleware (which fixes URL generation when the app is served behind a reverse proxy), then registers all ten blueprints. Running `python app.py` calls `create_app()` and then `app.run()`.

When adding a new blueprint, import it in `create_app()` and call `app.register_blueprint()`. Do not import blueprints at the module level of `app.py` before the factory is defined — this can cause circular import issues with blueprints that import from services.

### `config.py` — Configuration

All configuration values live here as dataclass fields. Add new fields here rather than reading environment variables directly in service code. The `get_config()` function is the single access point for the rest of the app. See [configuration.md](configuration.md) for a complete field reference.

### `models/` — Data Transfer Objects

The three model files define the data shapes that move between the service layer and route handlers:

- `chunk.py` — `Chunk`, `ChunkMetadata`, `ChunkType`
- `search_result.py` — `SearchResult`, `SearchResultSet`
- `query_history.py` — `QueryHistory`

Models are plain dataclasses. They do not call services, access the database, or perform I/O. Adding a new field to a model does not require any migration — ChromaDB stores metadata as a flat dict, and `ChunkMetadata.from_dict()` handles missing keys with sensible defaults.

### `routes/` — Request Handlers

Each file in `routes/` defines one Flask Blueprint. Route handlers should be short: validate the request, call a service, serialise the result. Business logic that appears in a route handler should be moved to a service.

The pattern for accessing services within route handlers is the lazy-initialisation singleton:

```python
_search_service: SearchService | None = None

def _get_search_service() -> SearchService:
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service

@search_bp.route("/api/search/semantic", methods=["POST"])
def semantic_search():
    svc = _get_search_service()
    ...
```

This pattern ensures that expensive service construction (embedding function setup, ChromaDB client) happens once and is shared across requests.

### `services/` — Business Logic

Services contain all the logic that is too complex for a route handler. Key services:

- `chroma_client.py` — `ChromaClientManager` singleton; never instantiate it directly, always use `get_chroma_client()`
- `ingestion_service.py` — `IngestionService`, `ASTParser`, `TextSplitter`, `MarkdownSplitter`
- `search_service.py` — `SearchService`, `SemanticSearchStrategy`, `RegexSearchStrategy`
- `visualization_service.py` — `VisualizationService`, `ReductionMethod`, `ColorScheme`

### `utils/` — Stateless Helpers

`text_splitter.py` contains lightweight display utilities that use a character-count heuristic instead of tiktoken. These are used in the UI layer where exact token counts are not needed and loading tiktoken would add latency.

`code_stats.py` contains functions for computing aggregate statistics from a list of chunks. These are pure functions with no side effects.

## Common Development Tasks

### Re-indexing After Ingestion Changes

If you change the ingestion pipeline (for example, adjusting token limits or adding a new file type), existing collections will be stale. Delete and re-create them:

1. Go to the Collections page in the UI
2. Delete the collection
3. Create a new collection with the same name and source directory

### Adding a New File Type

See the "Adding a New ChunkType" section in [CONTRIBUTING.md](../CONTRIBUTING.md) for the step-by-step checklist.

### Changing the Embedding Model

Update `ChromaConfig.embedding_model` in `config.py`. Then delete all existing collections and re-ingest from scratch — embedding vectors from different models are not comparable.

### Inspecting ChromaDB Data Directly

ChromaDB stores its data in the `persist_directory` (default `./chroma_data`). You can open a Python REPL and inspect it:

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_data")
coll = client.get_collection("my_project")
print(coll.count())
sample = coll.peek(5)
print(sample["metadatas"])
```

This is useful for debugging ingestion issues without starting the Flask server.

## Debugging Tips

### Checking Why a Search Returns Unexpected Results

1. Run the same query in the UI and note the top result's chunk ID
2. Use the Explorer to look up that chunk by ID and read the full content
3. Compare the chunk to your query — the semantic distance is computed between their embedding vectors, not their word overlap
4. If the result seems wrong, check whether the ingestion captured the expected content by searching for a distinctive symbol name in the regex search mode

### Diagnosing Slow Ingestion

Start the ingestion and watch the console output. Each batch of 100 chunks produces a progress update. If progress stalls on a specific file, that file is likely very large or contains a function that exceeds the token limit and takes many split iterations to process. Check the file size and consider excluding it with a path filter.
