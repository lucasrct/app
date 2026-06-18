"""Search service with semantic and regex strategies."""

import logging
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import chromadb

logger = logging.getLogger(__name__)

from models.chunk import Chunk, ChunkMetadata
from models.search_result import SearchResult, SearchResultSet
from config import get_config


class RegexTimeoutError(Exception):
    """Raised when a regex search exceeds its configured time limit."""


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    @abstractmethod
    def search(self, collection: chromadb.Collection, query: str,
               n_results: int = 10, filters: Optional[Dict] = None) -> SearchResultSet:
        """Execute a search and return results."""
        ...

    @abstractmethod
    def validate_query(self, query: str) -> tuple:
        """Validate a query string. Returns (is_valid, error_message)."""
        ...

    @staticmethod
    def _build_where_clause(filters: Dict) -> Optional[Dict]:
        """Build a ChromaDB where clause using only $eq operators (get/query compatible).

        path and symbol are excluded here — they require Python-side substring matching
        because ChromaDB's metadata where clause does not support $contains.
        """
        conditions = []
        if filters.get("chunk_type"):
            conditions.append({"chunk_type": {"$eq": filters["chunk_type"]}})
        for k, v in filters.items():
            if k not in {"path", "chunk_type", "symbol"} and v:
                conditions.append({k: {"$eq": str(v)}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _apply_substring_filters(results: list, filters: Dict) -> list:
        """Post-filter results for path and symbol substring matching."""
        path_sub = (filters.get("path") or "").lower()
        symbol_sub = (filters.get("symbol") or "").lower()
        if not path_sub and not symbol_sub:
            return results
        return [
            r for r in results
            if (not path_sub or path_sub in (r.chunk.metadata.path or "").lower())
            and (not symbol_sub or symbol_sub in (r.chunk.metadata.symbol or "").lower())
        ]


class SemanticSearchStrategy(SearchStrategy):
    """Dense/semantic search using ChromaDB's embedding-based query."""

    def validate_query(self, query: str) -> tuple:
        config = get_config().search
        if not query or not query.strip():
            return False, "Query cannot be empty"
        if len(query.strip()) < config.min_query_length:
            return False, f"Query must be at least {config.min_query_length} characters"
        if len(query.strip()) > config.max_query_length:
            return False, f"Query must be at most {config.max_query_length} characters"
        return True, ""

    def search(self, collection: chromadb.Collection, query: str,
               n_results: int = 10, filters: Optional[Dict] = None) -> SearchResultSet:
        logger.info(f"Semantic search: query='{query[:80]}' n_results={n_results} collection={collection.name}")
        start = time.time()

        where_clause = self._build_where_clause(filters) if filters else None

        query_kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, get_config().search.max_n_results),
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        raw = collection.query(**query_kwargs)
        elapsed_ms = (time.time() - start) * 1000

        results = []
        for i in range(len(raw["ids"][0])):
            chunk = Chunk.from_chroma_result(
                id=raw["ids"][0][i],
                document=raw["documents"][0][i],
                metadata=raw["metadatas"][0][i],
            )
            result = SearchResult(
                chunk=chunk,
                score=raw["distances"][0][i],
                rank=i + 1,
            )
            results.append(result)

        if filters:
            results = self._apply_substring_filters(results, filters)
        for i, r in enumerate(results):
            r.rank = i + 1

        logger.info(f"Semantic search completed: {len(results)} results in {elapsed_ms:.1f}ms")

        return SearchResultSet(
            results=results,
            query=query,
            total_time_ms=elapsed_ms,
            collection_name=collection.name,
        )


class RegexSearchStrategy(SearchStrategy):
    """Full-text regex search using ChromaDB's native where_document filter.

    Uses ChromaDB's $regex operator for server-side filtering,
    then extracts highlights and computes match-count scores client-side.
    See: https://docs.trychroma.com/docs/querying-collections/full-text-search
    """

    def validate_query(self, query: str) -> tuple:
        if not query or not query.strip():
            return False, "Pattern cannot be empty"
        try:
            re.compile(query)
        except re.error as e:
            return False, f"Invalid regex pattern: {e}"
        return True, ""

    def search(self, collection: chromadb.Collection, query: str,
               n_results: int = 50, filters: Optional[Dict] = None,
               timeout_seconds: Optional[float] = None) -> SearchResultSet:
        logger.info(f"Regex search: pattern='{query[:80]}' collection={collection.name}")
        start = time.time()

        try:
            pattern = re.compile(query, re.MULTILINE)
        except re.error as e:
            logger.warning(f"Invalid regex pattern: {e}")
            return SearchResultSet(query=query)

        where_clause = self._build_where_clause(filters) if filters else None
        get_kwargs: Dict[str, Any] = {
            "where_document": {"$regex": query},
            "include": ["documents", "metadatas"],
        }
        if where_clause:
            get_kwargs["where"] = where_clause

        timeout_s = timeout_seconds if timeout_seconds is not None else get_config().search.regex_timeout_seconds
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(collection.get, **get_kwargs)
            try:
                matched_data = future.result(timeout=timeout_s)
            except FuturesTimeoutError:
                raise RegexTimeoutError(
                    f"Regex search timed out after {timeout_s:.0f}s — "
                    "for a very broad pattern over a large codebase, consider narrowing the search instead of waiting. "
                    "Note that this is separate from the result cap, which limits how many matches a single search returns."
                )

        # Extract highlights and compute scores on the filtered results
        results = []
        for i in range(len(matched_data["ids"])):
            doc = matched_data["documents"][i]
            matches = list(pattern.finditer(doc))
            match_count = len(matches) if matches else 1

            chunk = Chunk.from_chroma_result(
                id=matched_data["ids"][i],
                document=doc,
                metadata=matched_data["metadatas"][i],
            )
            score = 1.0 / (1.0 + match_count)
            highlights = [m.group(0) for m in matches[:5]]
            results.append(SearchResult(
                chunk=chunk,
                score=score,
                highlights=highlights,
            ))

        if filters:
            results = self._apply_substring_filters(results, filters)

        # Sort by relevance (most matches first = lowest score)
        results.sort(key=lambda r: r.score)
        for i, r in enumerate(results[:n_results]):
            r.rank = i + 1

        elapsed_ms = (time.time() - start) * 1000

        return SearchResultSet(
            results=results[:n_results],
            query=query,
            total_time_ms=elapsed_ms,
            collection_name=collection.name,
        )


class MetadataSearchStrategy(SearchStrategy):
    """Pure metadata filter search — no query text, no embeddings.

    Uses ChromaDB's where clause to filter chunks by path, chunk_type,
    and/or symbol. Results are ordered by path then start_line.
    At least one filter must be provided.
    """

    def validate_query(self, query: str) -> tuple:
        return True, ""

    def search(self, collection: chromadb.Collection, query: str,
               n_results: int = 200, filters: Optional[Dict] = None) -> SearchResultSet:
        logger.info(f"Metadata search: filters={filters} collection={collection.name}")
        start = time.time()

        where_clause = self._build_where_clause(filters or {})

        get_kwargs: Dict[str, Any] = {"include": ["documents", "metadatas"]}
        if where_clause:
            get_kwargs["where"] = where_clause
        get_kwargs["limit"] = n_results

        data = collection.get(**get_kwargs)
        elapsed_ms = (time.time() - start) * 1000

        results = []
        for i in range(len(data["ids"])):
            chunk = Chunk.from_chroma_result(
                id=data["ids"][i],
                document=data["documents"][i],
                metadata=data["metadatas"][i],
            )
            results.append(SearchResult(chunk=chunk, score=0.0, rank=i + 1))

        if filters:
            results = self._apply_substring_filters(results, filters)

        # Sort by path then line number for predictable ordering
        results.sort(key=lambda r: (r.chunk.metadata.path, r.chunk.metadata.start_line))
        for i, r in enumerate(results):
            r.rank = i + 1

        logger.info(f"Metadata search: {len(results)} results in {elapsed_ms:.1f}ms")
        return SearchResultSet(
            results=results,
            query=query,
            total_time_ms=elapsed_ms,
            collection_name=collection.name,
        )


@dataclass
class SearchService:
    """Facade that dispatches to the appropriate search strategy."""
    semantic: SemanticSearchStrategy = None
    regex: RegexSearchStrategy = None
    metadata: MetadataSearchStrategy = None

    def __post_init__(self):
        if self.semantic is None:
            self.semantic = SemanticSearchStrategy()
        if self.regex is None:
            self.regex = RegexSearchStrategy()
        if self.metadata is None:
            self.metadata = MetadataSearchStrategy()

    def semantic_search(self, collection: chromadb.Collection, query: str,
                        n_results: int = 10, filters: Optional[Dict] = None) -> SearchResultSet:
        """Perform a semantic (dense embedding) search."""
        return self.semantic.search(collection, query, n_results, filters)

    def regex_search(self, collection: chromadb.Collection, pattern: str,
                     n_results: int = 50, filters: Optional[Dict] = None,
                     timeout_seconds: Optional[float] = None) -> SearchResultSet:
        """Perform a regex pattern search."""
        return self.regex.search(collection, pattern, n_results, filters, timeout_seconds)

    def metadata_search(self, collection: chromadb.Collection,
                        filters: Optional[Dict] = None,
                        n_results: int = 200) -> SearchResultSet:
        """Filter chunks by metadata fields without a query."""
        return self.metadata.search(collection, query="", filters=filters, n_results=n_results)

    def get_strategy(self, mode: str) -> SearchStrategy:
        """Get the search strategy by name."""
        strategies = {
            "semantic": self.semantic,
            "regex": self.regex,
            "metadata": self.metadata,
        }
        return strategies.get(mode, self.semantic)
