# Changelog

## [2.1.0] — 2024-11-20

### Added

- **Markdown and text ingestion**: The ingestion pipeline now supports `.md` and `.txt` files in addition to `.py`. Markdown files are split on H1–H3 headers, treating each section as an atomic chunk. Plain-text files are split by token budget using the existing line-accumulator splitter. Two new `ChunkType` values — `MARKDOWN_SECTION` and `TEXT_PARAGRAPH` — are used for metadata. The collection explorer and visualizer display them alongside code chunks without requiring UI changes.

- **`MarkdownSplitter` class**: Added to `services/ingestion_service.py`. Parses Markdown source into header-bounded sections and stores the header text as the chunk's `symbol` field. Oversized sections fall through to `TextSplitter` for token-bounded subdivision.

- **Extended `ChunkType` enum**: `MARKDOWN_SECTION = "markdown_section"` and `TEXT_PARAGRAPH = "text_paragraph"` added, each with a `display_label` and Bootstrap icon class for UI rendering.

### Changed

- `IngestionConfig.supported_extensions` extended from `(".py",)` to `(".py", ".md", ".txt")`.
- `IngestionService._discover_python_files()` renamed to `_discover_files()` to reflect that it now discovers all supported file types.
- `IngestionService.ingest_directory()` now dispatches to `chunk_file()` for `.py` and to `chunk_text_file()` for all other supported extensions.

### Fixed

- `TextSplitter.flush()` no longer emits an empty chunk when a file ends with a trailing newline. The flush now skips chunks whose document content is blank after stripping whitespace.

---

## [2.0.0] — 2024-09-15

### Breaking Changes

- The ingestion API endpoint changed from `POST /ingest` to `POST /api/collections` with a JSON body. The old endpoint is removed with no redirect.
- `ChunkMetadata` now requires a `language` field. Collections created with v1.x will return `"python"` as the default for existing chunks, but should be re-ingested for accurate language metadata.

### Added

- **Collection management UI**: Create, rename, and delete collections from the browser. Previously, collections could only be managed by calling the API directly.
- **Query history**: Every search is now logged with the query text, mode (semantic or regex), result count, and timestamp. The history panel shows the last 50 queries and allows replaying any of them with one click.
- **Similarity inspector**: Select any two chunks by ID and get their exact cosine similarity score. Useful for understanding why two chunks rank close together in search results.
- **Suggestions endpoint**: `GET /api/suggestions?q=term` returns up to five autocomplete suggestions based on symbol names in the active collection. The suggestions panel in the search UI is powered by this endpoint.

### Changed

- Upgraded ChromaDB from 0.4.x to 1.4.1. The `PersistentClient` API changed significantly; `Settings` objects are no longer passed to the client constructor.
- `SearchResultSet` now includes a `collection_name` field so the UI can display which collection produced the results.
- The embedding visualizer now defaults to PCA instead of random projection. Random projection is still available as an option.

### Fixed

- The regex search strategy no longer crashes when the pattern contains unescaped characters that are valid regex but produce zero matches. It now returns an empty `SearchResultSet` in that case.
- `VisualizationService` no longer raises `ValueError` when a collection contains fewer than two chunks, which previously caused PCA to fail. Collections with fewer than two chunks now return an empty point cloud.

---

## [1.1.0] — 2024-06-28

### Added

- **Regex tester route**: `POST /api/regex/test` accepts a pattern and a sample text and returns match positions and groups. This powers the live-preview panel in the search UI.
- **Code statistics endpoint**: `GET /api/statistics/{collection}` returns chunk counts by file, chunk type distribution, token size histogram, and the top 10 symbols by frequency.
- **Tutorial pages**: A five-page guided walkthrough explaining embeddings, vector databases, chunking strategies, and search modes. Accessible from the navigation bar.

### Changed

- `SemanticSearchStrategy` now caps `n_results` at the value set in `SearchConfig.max_n_results` (default 50) regardless of what the caller requests. Previously, requesting more results than are in the collection caused a ChromaDB error.

### Fixed

- The collection explorer no longer shows duplicate chunks when the same file is ingested twice without deleting the collection first. The UI now warns when the selected collection was last modified less than five minutes ago, suggesting a re-ingest may be in progress.

---

## [1.0.0] — 2024-04-02

### Initial Release

First public release of the code search app. Included:

- Flask application factory with ten blueprints
- AST-based Python ingestion using tree-sitter
- Semantic search via OpenAI embeddings and ChromaDB ANN
- Regex search via ChromaDB `$regex` operator
- Collection explorer with metadata filtering
- PCA-based embedding visualizer
- `ChromaClientManager` singleton for shared client and embedding function
