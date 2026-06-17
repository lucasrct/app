# API Reference

All endpoints return JSON. Error responses use the shape `{"error": "<message>"}` with an appropriate HTTP status code. Collection names must be non-empty strings containing only alphanumeric characters, underscores, and hyphens.

## Collections

### `POST /api/collections`

Create a new collection and optionally trigger ingestion immediately.

**Request body**

```json
{
  "name": "my_project",
  "source_dir": "/absolute/path/to/repo"
}
```

`source_dir` is optional. If omitted, an empty collection is created and ingestion must be triggered separately via `POST /api/collections/{name}/ingest`.

**Response** — `201 Created`

```json
{
  "name": "my_project",
  "chunk_count": 253,
  "created_at": "2024-11-20T14:32:00"
}
```

**Errors**: `400` if the name is invalid or already exists; `500` if ingestion fails.

---

### `GET /api/collections`

List all collections in the ChromaDB instance.

**Response** — `200 OK`

```json
{
  "collections": [
    {"name": "my_project", "chunk_count": 253},
    {"name": "other_repo", "chunk_count": 87}
  ]
}
```

---

### `DELETE /api/collections/{name}`

Delete a collection and all its chunks permanently.

**Response** — `200 OK` with `{"deleted": true}`. Returns `404` if the collection does not exist.

---

### `POST /api/collections/{name}/ingest`

Trigger ingestion of a source directory into an existing collection.

**Request body**

```json
{"source_dir": "/path/to/repo"}
```

**Response** — `200 OK` with a progress summary:

```json
{
  "total_files": 18,
  "processed_files": 18,
  "total_chunks": 261,
  "failed_files": [],
  "progress_percentage": 100.0,
  "is_complete": true
}
```

---

## Search

### `POST /api/search/semantic`

Run a semantic (dense embedding) search over a collection.

**Request body**

```json
{
  "collection": "my_project",
  "query": "how does the ingestion pipeline work",
  "n_results": 10,
  "filters": {
    "path": "services",
    "chunk_type": "function_definition"
  }
}
```

`filters` is optional. Each filter field is optional and uses substring matching for `path`, exact match for `chunk_type` and `symbol`.

**Response** — `200 OK`

```json
{
  "results": [
    {
      "rank": 1,
      "score": 0.3821,
      "chunk": {
        "id": "abc123",
        "document": "def chunk_file(self, file_path ...",
        "metadata": {
          "path": "services/ingestion_service.py",
          "start_line": 173,
          "end_line": 253,
          "symbol": "chunk_file",
          "chunk_type": "function_definition",
          "language": "python"
        }
      }
    }
  ],
  "query": "how does the ingestion pipeline work",
  "total_time_ms": 214.5,
  "collection_name": "my_project"
}
```

**Errors**: `400` if query is empty or shorter than the configured minimum length; `404` if the collection does not exist.

---

### `POST /api/search/regex`

Search by regular expression pattern applied to chunk text.

**Request body**

```json
{
  "collection": "my_project",
  "pattern": "def\\s+chunk_\\w+",
  "n_results": 50
}
```

**Response** — same shape as semantic search, with an additional `highlights` list per result containing the matched substrings (up to five per chunk).

**Errors**: `400` if the pattern is not valid regex; `404` if the collection does not exist.

---

## Explorer

### `GET /api/explorer/{collection}`

Browse chunks in a collection with optional metadata filters.

**Query parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Filter by file path substring |
| `chunk_type` | string | Exact match on chunk type |
| `symbol` | string | Exact match on symbol name |
| `page` | int | Page number (1-indexed, default 1) |
| `per_page` | int | Results per page (default 20, max 100) |

**Response** — `200 OK` with a `chunks` array and pagination metadata.

---

## Visualizer

### `POST /api/visualizer/points`

Generate a 2D point cloud from a collection's embeddings.

**Request body**

```json
{
  "collection": "my_project",
  "method": "pca",
  "color_by": "type",
  "max_points": 200
}
```

`method` is `"pca"` or `"random_projection"`. `color_by` is `"file"`, `"type"`, or `"symbol"`.

**Response** — `200 OK` with a `points` array of `{x, y, label, color, id}` objects.

---

### `POST /api/visualizer/chunks`

Fetch full chunk content for a list of chunk IDs (used when clicking points in the scatter plot).

**Request body**

```json
{
  "collection": "my_project",
  "chunk_ids": ["abc123", "def456"]
}
```

**Response** — `200 OK` with a `chunks` array containing `id`, `code`, `path`, `chunk_type`, and `symbol` for each requested chunk.

---

## Statistics

### `GET /api/statistics/{collection}`

Return aggregate metrics for a collection.

**Response** — `200 OK`

```json
{
  "total_chunks": 253,
  "by_type": {
    "function_definition": 180,
    "class_definition": 32,
    "gap": 38,
    "markdown_section": 3
  },
  "by_file": {"services/ingestion_service.py": 42, ...},
  "top_symbols": [["chunk_file", 1], ["search", 2], ...],
  "token_distribution": {"p50": 120, "p90": 480, "p99": 950, "max": 998}
}
```

---

## Regex Tester

### `POST /api/regex/test`

Validate a regex pattern and test it against sample text without querying the collection.

**Request body**

```json
{
  "pattern": "def\\s+\\w+",
  "text": "def my_function():\n    pass"
}
```

**Response** — `200 OK` with `{"valid": true, "matches": [{"start": 0, "end": 14, "text": "def my_function"}]}`. If the pattern is invalid, returns `{"valid": false, "error": "..."}`.
