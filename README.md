# ChromaDB Code Search UI

A Flask web application for searching, browsing, and visualizing Python code using semantic embeddings and ChromaDB. Built as a companion app for the **Context Engineering with Chroma** course.

## Purpose

This app serves two roles in the course:

1. **Teaching material** — Students ingest this codebase into ChromaDB using the AST-based chunking pipeline they build in the labs. The well-structured Python code (models, services, routes, utils) makes it an ideal target for practicing chunking strategies.
2. **Interactive tool** — Once ingested, students launch this app to explore their collections, run searches, and see how their chunking and metadata decisions affect retrieval quality.

## Features

- **Semantic search** — Natural language queries over code using OpenAI embeddings (`text-embedding-3-small`)
- **Regex search** — Structural pattern matching across the codebase with analysis and explanation
- **Collection explorer** — Paginated chunk browser with filters by file path, chunk type, and symbol name
- **Code statistics** — Construct detection, size distributions, and symbol rankings
- **Embedding visualizer** — 2-D projection of chunk embeddings to explore clustering
- **Smart suggestions** — Context-aware query suggestions based on collection metadata
- **Query history and bookmarks** — Persistent search history with color-coded bookmarks
- **Interactive tutorials** — Guided tours with spotlight overlays for onboarding

## How Search Works

The app searches the indexed code in two complementary ways.

**Semantic search** ranks code chunks by how close they are to your query in meaning, using the same embedding model that indexed them. It finds relevant code even when your wording does not match the wording in the code.

**Regex search** matches an exact pattern across the chunks. It is the right tool for a specific identifier, a function call, or a structural pattern that semantic search would only approximate.

### Searching the collection

Semantic search ranks code chunks by how close they are to your query.
By default it returns the top 10 matches, and you can ask for as many as 50.
A query must be at least 2 characters, and no more than 500.

Every result carries its chunk's metadata — the file path, the line range, and the symbol name — so you can trace a match back to its place in the source.

## How Ingestion Works

Before anything can be searched, the code has to be ingested into a ChromaDB collection. The pipeline walks the repository, reads each Python file, and splits it into chunks along its structure rather than by raw length.

It uses **tree-sitter** to parse each file into an abstract syntax tree, then takes whole **functions** and **classes** as atomic chunks. Code that falls between definitions, such as imports and top-level statements, becomes its own "gap" chunk, so nothing in the file is dropped. A chunk that would exceed the embedding model's token limit of 1000 tokens is split further along line boundaries.

Each chunk is stored with metadata: its file path, its start and end line, its symbol name, and its chunk type (function, class, module, or gap). Only Python files are ingested.

## Project Structure

```
app/
├── app.py                  # Flask application factory and entry point
├── config.py               # Dataclass-based configuration (env vars, defaults)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── models/                 # Data models
│   ├── chunk.py            # Chunk, ChunkMetadata, ChunkType
│   ├── search_result.py    # SearchResult, SearchResultSet, ResultFormatter
│   └── query_history.py    # QueryRecord, Bookmark, HistoryManager
│
├── routes/                 # Flask blueprints (one per feature)
│   ├── search.py           # Semantic and regex search endpoints
│   ├── collections.py      # Collection CRUD and ingestion triggers
│   ├── explorer.py         # Paginated chunk browsing with filters
│   ├── similarity.py       # Pairwise similarity matrix computation
│   ├── history.py          # Query history and bookmarks API
│   ├── regex_tester.py     # Regex testing and analysis
│   ├── suggestions.py      # Smart query suggestions
│   ├── statistics.py       # Code metrics and analytics
│   ├── visualizer.py       # 2-D embedding visualization
│   └── tutorial.py         # Interactive guided tours
│
├── services/               # Business logic layer
│   ├── chroma_client.py    # ChromaDB connection manager (singleton)
│   ├── search_service.py   # Search strategies (semantic + regex)
│   ├── collection_service.py   # Collection management and stats
│   ├── ingestion_service.py    # AST parsing and code chunking pipeline
│   ├── similarity_service.py   # Vector similarity computations
│   ├── statistics_service.py   # Code metrics and analysis
│   ├── visualization_service.py # Dimensionality reduction for the 2-D view
│   ├── suggestion_service.py   # Multi-strategy suggestion generator
│   └── tutorial_service.py     # Tutorial builder and manager
│
├── utils/                  # Utilities and helpers
│   ├── validators.py       # Input validation (queries, paths, regex)
│   ├── regex_engine.py     # Regex analysis and human-readable explanation
│   ├── code_parser.py      # Lightweight regex-based Python parser
│   ├── text_splitter.py    # Token-based text splitting
│   └── formatters.py       # Display formatting (scores, code, paths)
│
├── templates/              # Jinja2 HTML templates
│   ├── base.html           # Base layout with navbar and tutorial engine
│   ├── index.html          # Dashboard (collection cards)
│   ├── search.html         # Search interface
│   ├── explorer.html       # Chunk browser
│   └── collection.html     # Collection detail page
│
└── static/
    └── css/style.css       # Custom styles
```

## Design Patterns

The codebase intentionally demonstrates several software design patterns, making it a richer target for code search exercises:

- **Strategy** — `SearchStrategy`, `SimilarityComputer`, `DimensionReducer`, `SuggestionStrategy`
- **Singleton** — `ChromaClientManager` for a single DB connection
- **Factory** — `get_reducer()`, `get_similarity_computer()`, `get_tutorial_builder()`
- **Builder** — Tutorial builders (`DashboardTutorialBuilder`, `CollectionTutorialBuilder`)
- **Facade** — `SearchService`, `SuggestionService`, `StatisticsService` wrapping multiple strategies

## Configuration

The app reads its settings from `config.py`, overridable by environment variables. The defaults that matter most:

| Setting | Default | Meaning |
|---------|---------|---------|
| Embedding model | `text-embedding-3-small` | model used to embed code and queries |
| Default results | 10 | matches returned per search |
| Maximum results | 50 | upper limit on matches per search |
| Query length | 2–500 characters | accepted query size |
| Max tokens per chunk | 1000 | a larger chunk is split along line boundaries |
| Regex result cap | 100 | maximum matches a regex search returns |
| Regex timeout | 5 seconds | a regex search is abandoned after this |

## Limitations

A few constraints are worth knowing before you rely on the app:

- **Python only.** Ingestion handles `.py` files; other languages are skipped.
- **Regex search is bounded.** It returns at most 100 matches and times out after 5 seconds, so a very broad pattern may not surface everything.
- **The embedding view is a projection.** The 2-D visualizer reduces high-dimensional embeddings down to two dimensions, which is useful for spotting clusters but loses detail.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables (copy `.env.example` to `.env`):
   ```
   OPENAI_API_KEY=sk-your-key-here
   CHROMA_PERSIST_DIR=./chroma_data
   ```

3. Run the app:
   ```bash
   python app.py
   ```

## Dependencies

| Package | Purpose |
|---------|---------|
| flask | Web framework |
| chromadb | Vector database |
| openai | Embedding API |
| tiktoken | Token counting |
| tree-sitter | AST parsing |
| tree-sitter-python | Python grammar for tree-sitter |
| python-dotenv | Environment variable management |
| pathspec | `.gitignore` pattern matching |
