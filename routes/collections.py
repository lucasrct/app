"""Collection management routes."""

import os
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash

from services.collection_service import CollectionService
from services.ingestion_service import IngestionService
from utils.validators import validate_collection_name, validate_directory_path

collections_bp = Blueprint("collections", __name__)

_collection_service = None
_ingestion_service = None


def _get_collection_service():
    global _collection_service
    if _collection_service is None:
        _collection_service = CollectionService()
    return _collection_service


def _get_ingestion_service():
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service


@collections_bp.route("/")
def index():
    """Dashboard: show all collections with stats."""
    collections = _get_collection_service().list_collections()
    return render_template("index.html",
                           collections=collections,
                           active_page="dashboard")


@collections_bp.route("/collection/<name>")
def collection_detail(name: str):
    """Show detailed stats for a single collection."""
    stats = _get_collection_service().get_collection_stats(name)
    if stats is None:
        flash(f"Collection '{name}' not found.", "danger")
        return redirect(url_for("collections.index"))
    return render_template("collection.html",
                           stats=stats,
                           active_page="collections")


@collections_bp.route("/api/collections", methods=["POST"])
def api_create_collection():
    """API: Create a new collection."""
    data = request.get_json() or {}
    name = data.get("name", "")
    embedding_provider = data.get("embedding_provider", "openai")
    embedding_model = data.get("embedding_model", "text-embedding-3-small")

    is_valid, error = validate_collection_name(name)
    if not is_valid:
        return jsonify({"error": error}), 400

    success = _get_collection_service().create_collection(
        name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
    if success:
        return jsonify({"message": f"Collection '{name}' created", "name": name})
    return jsonify({"error": "Failed to create collection"}), 500


@collections_bp.route("/api/collections/<name>", methods=["DELETE"])
def api_delete_collection(name: str):
    """API: Delete a collection."""
    success = _get_collection_service().delete_collection(name)
    if success:
        return jsonify({"message": f"Collection '{name}' deleted"})
    return jsonify({"error": f"Failed to delete '{name}'"}), 500


@collections_bp.route("/api/collections/<name>/ingest", methods=["POST"])
def api_ingest_directory(name: str):
    """API: Ingest a directory into a collection."""
    data = request.get_json()
    directory = data.get("directory", "")

    is_valid, error = validate_directory_path(directory)
    if not is_valid:
        return jsonify({"error": error}), 400

    progress = _get_ingestion_service().ingest_directory(directory, name)

    return jsonify({
        "message": "Ingestion complete",
        "progress": progress.to_dict(),
    })


@collections_bp.route("/api/json_preview")
def api_json_preview():
    """API: Read the first record of a JSON file and return its fields + sample values."""
    import json as _json
    path = request.args.get("path", "").strip()
    if not path:
        return jsonify({"error": "path is required"}), 400
    if not os.path.isfile(path):
        return jsonify({"error": f"File not found: {path}"}), 400
    if not path.lower().endswith((".json", ".jsonl")):
        return jsonify({"error": "File must be .json or .jsonl"}), 400

    try:
        from services.ingestion_service import IngestionService
        records = IngestionService._load_records(path)

        if not records:
            return jsonify({"error": "File contains no records"}), 400

        first = records[0]
        fields = [
            {"name": k, "sample": (str(v)[:100] if v is not None else "(null)")}
            for k, v in first.items()
        ]
        return jsonify({"fields": fields, "total_records": len(records)})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@collections_bp.route("/api/browse")
def api_browse():
    """API: List directory contents for the file browser."""
    path = request.args.get("path", "").strip()
    if not path:
        path = os.path.expanduser("~")

    path = os.path.realpath(path)

    if not os.path.isdir(path):
        parent = os.path.dirname(path)
        if os.path.isdir(parent):
            path = parent
        else:
            path = os.path.expanduser("~")

    try:
        entries = []
        for entry_name in sorted(os.listdir(path), key=lambda n: (not os.path.isdir(os.path.join(path, n)), n.lower())):
            if entry_name.startswith("."):
                continue
            full = os.path.join(path, entry_name)
            if os.path.isdir(full):
                entries.append({"name": entry_name, "path": full, "type": "dir"})
            elif entry_name.lower().endswith((".json", ".jsonl")):
                entries.append({"name": entry_name, "path": full, "type": "json"})

        parent = os.path.dirname(path)
        return jsonify({
            "current": path,
            "parent": parent if parent != path else None,
            "entries": entries,
        })
    except PermissionError:
        return jsonify({"error": "Permission denied reading that directory"}), 403


@collections_bp.route("/api/collections/<name>/metadata_keys")
def api_metadata_keys(name: str):
    """API: Return the unique metadata keys present in a collection (sampled)."""
    try:
        from services.chroma_client import get_chroma_client
        manager = get_chroma_client()
        collection = manager.get_existing_collection(name)
        if collection is None:
            return jsonify({"error": f"Collection not found: {name}"}), 404
        sample = collection.get(limit=50, include=["metadatas"])
        keys: set = set()
        for meta in (sample.get("metadatas") or []):
            if meta:
                keys.update(meta.keys())
        keys.discard("ingested_at")
        return jsonify({"keys": sorted(keys)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@collections_bp.route("/api/collections/<name>/ingest_json", methods=["POST"])
def api_ingest_json(name: str):
    """API: Ingest a JSON file into a collection."""
    data = request.get_json()
    file_path = (data.get("file_path") or "").strip()
    text_field = (data.get("text_field") or "text").strip() or "text"
    id_field = (data.get("id_field") or "").strip() or None
    metadata_fields = data.get("metadata_fields") or None  # list[str] or None

    if not file_path:
        return jsonify({"error": "file_path is required"}), 400
    if not os.path.isfile(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 400
    if not file_path.lower().endswith((".json", ".jsonl")):
        return jsonify({"error": "Only .json or .jsonl files are supported"}), 400

    try:
        result = _get_ingestion_service().ingest_json_file(
            file_path=file_path,
            collection_name=name,
            text_field=text_field,
            id_field=id_field,
            metadata_fields=metadata_fields,
        )
        return jsonify({"message": "JSON ingestion complete", "result": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Ingestion failed: {e}"}), 500
