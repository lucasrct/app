# Contributing

This document describes the conventions and processes for contributing to the code search app.

## Development Environment

Python 3.10 or higher is required. Clone the repo, install dependencies, and set your OpenAI API key:

```bash
cd app
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
python app.py
```

For development with auto-reload:

```bash
FLASK_ENV=development python app.py
```

Flask's built-in reloader restarts the server whenever a Python file changes. The ChromaDB client reconnects automatically on restart, but any in-memory state (query history, cached services) is reset.

## Code Style

The project does not yet enforce a formatter automatically. Follow these conventions manually until a CI step is added:

- 4-space indentation, no tabs
- Maximum line length of 100 characters
- Single blank line between methods; two blank lines between top-level definitions
- `snake_case` for functions, variables, and module names
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for module-level constants

All public functions and methods must have Google-style docstrings:

```python
def chunk_file(self, file_path: str, max_tokens: int = None) -> list[Chunk]:
    """Chunk a single Python file using hybrid AST and token splitting.

    Args:
        file_path: Absolute or relative path to the Python source file.
        max_tokens: Override for the configured per-chunk token limit.

    Returns:
        List of Chunk objects ready for upload to ChromaDB.

    Raises:
        FileNotFoundError: If file_path does not exist.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
```

## Adding a New Route

Each route module is a Flask Blueprint registered in `app.py`. To add a new feature:

1. Create a new file in `routes/` named after the feature, for example `routes/export.py`.
2. Define a Blueprint at the top of the file:
   ```python
   export_bp = Blueprint("export", __name__)
   ```
3. Add route functions decorated with `@export_bp.route(...)`.
4. Import and register the blueprint in `app.py`:
   ```python
   from routes.export import export_bp
   app.register_blueprint(export_bp)
   ```

Route handlers should validate input, call into a service, and return a JSON response. Business logic belongs in services, not in route handlers.

## Adding a New Service

Services are instantiated lazily using module-level globals. The pattern used throughout the codebase is:

```python
_my_service: MyService | None = None

def _get_my_service() -> MyService:
    global _my_service
    if _my_service is None:
        _my_service = MyService()
    return _my_service
```

Call `_get_my_service()` inside each route handler that needs it. Do not instantiate services at import time, as this would trigger ChromaDB and OpenAI initialisation before the application is configured.

## Adding a New ChunkType

If the ingestion pipeline gains support for a new file format:

1. Add a new member to the `ChunkType` enum in `models/chunk.py`:
   ```python
   MY_TYPE = "my_type"
   ```
2. Add a `display_label` entry in the `display_label` property.
3. Add an `icon` entry in the `icon` property (use a Bootstrap Icons class name).
4. Implement a splitter class in `services/ingestion_service.py`.
5. Add the file extension to `IngestionConfig.supported_extensions` in `config.py`.
6. Add a dispatch branch in `IngestionService.ingest_directory()`.

## Testing

There is no automated test suite at present. To verify changes manually:

1. Start the app with `python app.py`.
2. Ingest the `app/` directory itself using the Collections UI.
3. Run a semantic search for a feature you expect to find (for example, "how does regex search work").
4. Verify the top results are from `services/search_service.py` or related files.
5. Run a regex search for a pattern you know exists (for example, `class\s+\w+Strategy`).
6. Open the embedding visualizer and confirm points render without errors.

## Pull Request Guidelines

- Keep each pull request focused on a single concern. A PR that adds a new route and refactors the ingestion service at the same time is harder to review.
- Update `CHANGELOG.md` under an `[Unreleased]` section before opening the PR.
- Run the manual verification steps above before marking the PR as ready for review.
- If your change affects the ingestion pipeline, re-ingest the `app/` directory and confirm the collection statistics look correct (chunk count, symbol count, file count).

## Reporting Issues

Open a GitHub issue with:

- The Python version and OS
- The exact command that triggered the bug
- The full error message and traceback
- What you expected to happen vs. what actually happened
