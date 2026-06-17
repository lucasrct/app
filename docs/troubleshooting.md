# Troubleshooting

## Startup Errors

### `ValueError: The OPENAI_API_KEY environment variable is not set.`

**Cause**: The `OpenAIEmbeddingFunction` initialised by `ChromaClientManager` requires an OpenAI API key.

**Fix**: Set the environment variable before starting the server:

```bash
export OPENAI_API_KEY=sk-your-key-here
python app.py
```

If you do not have an OpenAI key and want to use ChromaDB's built-in embedding model instead, change `embedding_model` in `ChromaConfig` to `EmbeddingModel.DEFAULT`. Note that the default model runs locally and produces different embeddings — any existing collections must be re-ingested.

---

### `ModuleNotFoundError: No module named 'chromadb'`

**Cause**: Dependencies are not installed.

**Fix**: Run `pip install -r requirements.txt` from the `app/` directory.

---

### `Address already in use (port 5000)`

**Cause**: Another process is using port 5000.

**Fix**: Kill the other process or change the port in `config.py` (`AppConfig.port`). On macOS, port 5000 is used by AirPlay Receiver in System Preferences.

---

## Ingestion Errors

### Ingestion completes but collection shows 0 chunks

**Cause A**: The `source_dir` path does not exist or contains no supported files.

**Fix**: Confirm the directory path and that it contains `.py`, `.md`, or `.txt` files. Use an absolute path to avoid working-directory ambiguity.

**Cause B**: All files were skipped because their extensions are not in `IngestionConfig.supported_extensions`.

**Fix**: Check `config.py` and verify the extensions you need are present in the tuple.

---

### `UnicodeDecodeError` during ingestion

**Cause**: A `.md` or `.txt` file is not valid UTF-8 (for example, a file with Windows-1252 encoding or binary content with a `.txt` extension).

**Fix**: The `chunk_text_file` method opens files with `encoding="utf-8"`. Add an `errors="replace"` argument to the `open()` call if you need to tolerate non-UTF-8 content, or add the problematic file to `ignore_patterns`.

---

### Failed files list is non-empty after ingestion

The `ingest_directory` method catches per-file exceptions and records them in `progress.failed_files` rather than aborting the whole batch. Check the response from `POST /api/collections/{name}/ingest` for the `failed_files` list. Each entry is a string of the form `"path/to/file.py: <error message>"`.

Common causes:

- **Syntax errors in Python files**: tree-sitter will still parse them, but may produce unexpected node boundaries
- **Binary files with supported extensions**: unlikely, but possible if a repository contains compiled `.py` files
- **Permission errors**: the process cannot read a file

---

## Search Errors

### Semantic search returns results that seem unrelated to the query

**Cause A**: The collection was ingested with a different embedding model than the one currently configured. Vectors from different models are not comparable.

**Fix**: Delete the collection and re-ingest with the current model.

**Cause B**: The query is too short or too generic. Single-word queries often match chunks that mention the word in a comment rather than chunks that implement the concept.

**Fix**: Use more descriptive queries ("how does chunking handle oversized functions" rather than "chunking").

---

### `404 Collection not found: my_collection`

**Cause**: The collection was deleted, or the `persist_directory` was changed and the collection was created under a different path.

**Fix**: List available collections with `GET /api/collections` and verify the name. If the `chroma_data/` directory has been moved or re-created, re-ingest to rebuild the collection.

---

### Regex search returns no results for a pattern I know exists

**Cause A**: The `$regex` operator in ChromaDB uses RE2 syntax, not Python's `re` module. Some Python regex features (lookaheads, lookbehinds, backreferences) are not supported.

**Fix**: Simplify the pattern to avoid unsupported features. Test the pattern against sample text using `POST /api/regex/test` before running it on the full collection.

**Cause B**: The matching text was split across chunk boundaries during ingestion. If a function definition spans more than 1,000 tokens, the chunk containing the relevant code may not include the line you are searching for.

**Fix**: Use the Explorer to search by symbol name and read the full chunk to see where the split occurred.

---

## Visualizer Errors

### Scatter plot renders empty or throws a 500 error

**Cause A**: The collection has fewer than two chunks, which makes PCA undefined.

**Fix**: Ingest more files or switch to the `random_projection` method, which handles small datasets more gracefully.

**Cause B**: `scikit-learn` is not installed.

**Fix**: Run `pip install scikit-learn`.

---

### Points in the scatter plot do not cluster by file as expected

This is expected behaviour for collections where most chunks come from a single file. PCA and random projection separate points by their embedding distance, not by file membership. If the code across files is semantically similar, the points will intermingle regardless of their file of origin. Try colouring by `chunk_type` instead of `file` to see structural differences.
