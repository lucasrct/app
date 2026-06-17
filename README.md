# Code Search App

A Flask-based web application for indexing, searching, and exploring Python codebases stored in ChromaDB. The app ingests a source directory, splits the code into semantic chunks using Abstract Syntax Trees and token-bounded splitting, and provides a browser UI for semantic and regex search over the resulting collection.

## Overview

When you point the app at a source directory, it walks every supported file, parses Python files using tree-sitter to extract function and class definitions as atomic units, and stores each chunk alongside rich metadata (file path, line range, symbol name, chunk type) in a ChromaDB persistent collection. You can then search the indexed codebase using natural-language queries (semantic search via OpenAI embeddings) or regular expressions (full-text filter via ChromaDB's `$regex` operator).

The same codebase that powers the app is also a valid target for indexing, which lets you point the ingestion pipeline at the `app/` directory itself and explore its own architecture through the search UI.

## Features

- **Semantic search**: Embed a natural-language query and retrieve the most similar code chunks using cosine distance over OpenAI `text-embedding-3-small` vectors.
- **Regex search**: Filter the collection by a regular expression pattern applied to chunk text. Useful for finding all usages of a specific function, class, or import.
- **Collection explorer**: Browse all ingested chunks with filters by file path, chunk type (function, class, top-level code), and symbol name.
- **Embedding visualizer**: Project chunk embeddings into two dimensions using PCA or random projection and explore the resulting scatter plot coloured by file, chunk type, or symbol.
- **Similarity inspector**: Select any two chunks and compare their cosine similarity score directly.
- **Code statistics**: View aggregate metrics — chunk count by file, token distribution, top symbols by frequency.
- **Query history**: Every search is logged with its query, mode, and timestamp so you can replay or compare previous searches.
- **Tutorial**: Built-in guided walkthrough explaining embeddings, chunking, and search modes.

## Setup

### Requirements

- Python 3.10 or higher
- An `OPENAI_API_KEY` environment variable set to a valid OpenAI API key

### Installation

```bash
cd app
pip install -r requirements.txt
```

### Running

```bash
cd app
OPENAI_API_KEY=your-key python app.py
```

The app starts on `http://localhost:5000` by default. Navigate to that URL in a browser to access the UI.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Used to generate embeddings via `text-embedding-3-small` |
| `CHROMA_PERSIST_DIR` | No | Directory for ChromaDB storage (default: `./chroma_data`) |
| `FLASK_ENV` | No | Set to `development` for debug mode (default: `development`) |
| `SECRET_KEY` | No | Flask session secret (use a random string in production) |

## Project Layout

```
app/
├── app.py                   # Flask application factory
├── config.py                # AppConfig, ChromaConfig, IngestionConfig, SearchConfig
├── requirements.txt
├── models/
│   ├── chunk.py             # Chunk, ChunkMetadata, ChunkType
│   ├── search_result.py     # SearchResult, SearchResultSet
│   └── query_history.py     # QueryHistory entry model
├── routes/
│   ├── collections.py       # Collection CRUD endpoints
│   ├── explorer.py          # Chunk browsing endpoints
│   ├── search.py            # Semantic and regex search endpoints
│   ├── visualizer.py        # Embedding projection endpoints
│   ├── similarity.py        # Pairwise similarity endpoints
│   ├── statistics.py        # Aggregate metrics endpoints
│   ├── history.py           # Query history endpoints
│   ├── suggestions.py       # Search suggestion endpoints
│   ├── regex_tester.py      # Live regex validation endpoint
│   └── tutorial.py          # Tutorial page routes
├── services/
│   ├── chroma_client.py     # Singleton ChromaDB client manager
│   ├── ingestion_service.py # AST parsing, text splitting, batch upload
│   ├── search_service.py    # SemanticSearchStrategy, RegexSearchStrategy
│   └── visualization_service.py  # PCA / random-projection dimensionality reduction
└── utils/
    ├── text_splitter.py     # Lightweight token estimation utilities
    └── code_stats.py        # Chunk aggregation and statistics helpers
```

## Ingesting a Codebase

Use the Collections UI to create a new collection, then trigger ingestion on any local directory. Alternatively, call the ingestion API directly:

```bash
curl -X POST http://localhost:5000/api/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my_project", "source_dir": "/path/to/repo"}'
```

Progress is streamed back as server-sent events. Once complete, the collection is searchable immediately.
