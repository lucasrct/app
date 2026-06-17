# Architecture

## Overview

The code search app is a Flask application built around three core operations: ingestion, storage, and retrieval. A source directory is parsed into chunks, the chunks are embedded and stored in ChromaDB, and the search UI queries the collection at read time. Each operation is handled by a dedicated service class with no cross-service coupling except through explicit constructor injection.

The app follows a blueprint-based routing structure with ten route modules. Each blueprint is self-contained and imports only from the service layer, never from other route modules. Services are instantiated lazily on first request using module-level globals and a getter function pattern.

## Request Lifecycle

A typical semantic search request follows this path:

1. The browser POSTs a JSON body to `/api/search/semantic`
2. The `search` blueprint receives the request and calls `_get_search_service()`
3. `_get_search_service()` returns a cached `SearchService` singleton (constructing one on first call)
4. `SearchService.semantic_search()` delegates to `SemanticSearchStrategy.search()`
5. `SemanticSearchStrategy` calls `collection.query(query_texts=[query], n_results=n)` on ChromaDB
6. ChromaDB embeds the query text using the attached `OpenAIEmbeddingFunction` and runs an ANN search
7. Results come back as raw dicts; they are converted to `Chunk` and `SearchResult` objects
8. The route handler serialises the `SearchResultSet` to JSON and returns it
9. The browser renders the results without a page reload

## Ingestion Pipeline

The ingestion pipeline lives in `services/ingestion_service.py` and is composed of three classes:

**`ASTParser`** wraps tree-sitter to produce a flat list of target nodes (function and class definitions) from a Python source file. It stops recursion at the first matching node, so a class is returned as a single atomic unit that includes all of its methods.

**`TextSplitter`** takes any string and a token budget and produces a list of line-bounded chunks that each fit within that budget. It is used for two purposes: splitting the gaps between AST nodes (imports, global variables, module-level comments) and splitting AST nodes that are too large to embed in a single call.

**`MarkdownSplitter`** handles `.md` files. It splits on H1–H3 headers, treating each header and its following content as one section. Oversized sections are passed to `TextSplitter` as a fallback.

**`IngestionService`** orchestrates the pipeline: it walks the source directory, dispatches each file to either `chunk_file()` (for Python) or `chunk_text_file()` (for Markdown and plain text), buffers chunks, and uploads them to ChromaDB in batches of 100.

## ChromaDB Client

`services/chroma_client.py` manages the connection to a ChromaDB `PersistentClient`. A `ChromaClientManager` singleton holds the client instance and the `OpenAIEmbeddingFunction`. The singleton is exposed through a module-level `get_chroma_client()` function that constructs it on first call and caches it for the lifetime of the process.

Collections are created with the embedding function attached, so ChromaDB automatically embeds any document text passed to `collection.add()` and any query text passed to `collection.query()`. The app never calls the embedding API directly.

## Search Strategies

Two strategies are available and share the `SearchStrategy` abstract base class:

**`SemanticSearchStrategy`** uses `collection.query()` with `query_texts`. The distance metric is cosine similarity (ChromaDB's default for OpenAI embedding functions). Results are ranked by distance ascending, so lower scores mean higher similarity.

**`RegexSearchStrategy`** uses `collection.get()` with a `where_document={"$regex": pattern}` filter. ChromaDB evaluates the regex server-side, which avoids pulling all documents into Python memory. The strategy then computes a match-count score client-side and sorts by it.

Both strategies accept an optional `filters` dict that maps to ChromaDB metadata `where` clauses (filtering by `path`, `chunk_type`, or `symbol`).

## Visualization Service

`services/visualization_service.py` reduces high-dimensional embeddings to two dimensions for the scatter-plot UI. It retrieves all embeddings from the collection via `collection.get(include=["embeddings", "metadatas"])`, applies either PCA (`sklearn.decomposition.PCA`) or random projection (`sklearn.random_projection.GaussianRandomProjection`), and returns a `PointCloud` dataclass holding 2D coordinates alongside colour-coding metadata.

The dimensionality reduction happens on every request because storing pre-reduced coordinates would go stale whenever the collection changes. For collections with more than 500 points, the service samples down to a configurable maximum before reducing.

## Configuration

All configuration is centralised in `config.py` as a tree of frozen dataclasses: `AppConfig` → `ChromaConfig`, `SearchConfig`, `IngestionConfig`. The `get_config()` factory reads environment variables once and returns the populated config object. Services access config through `get_config()` at construction time, not at request time.

## Blueprint Registration

The Flask application factory (`app.py`) registers all ten blueprints with URL prefixes under `/api/` for data endpoints and `/` for page endpoints. Blueprints that serve HTML pages use Jinja2 templates from the `templates/` directory. Blueprints that serve JSON data use `flask.jsonify`. There is no GraphQL layer or WebSocket connection.

## Key Design Constraints

**Singleton services**: Route handlers must not construct new `SearchService` or `IngestionService` instances per request. Both are expensive to initialise (embedding function setup, ChromaDB connection). The module-level lazy-initialisation pattern enforces this without a dependency injection framework.

**No ORM**: The app stores all persistent state in ChromaDB. There is no SQL database, no migration system, and no ORM. Query history is stored in a module-level list and is not durable across restarts.

**Stateless routes**: Route handlers perform no side-effectful caching. State lives in services and in ChromaDB. This makes routes easy to test by mocking the service layer.
