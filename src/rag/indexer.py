"""Persistent search indexer with incremental updates and hybrid representations."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.rag.config import RAGConfig, load_rag_config
from src.rag.knowledge_base import DocumentChunk, KnowledgeBase

logger = logging.getLogger("rag_indexer")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words/terms, handling Vietnamese, English, and code symbols."""
    # Keep Unicode words intact: the old regex split Vietnamese words at accents.
    stop = {'the','and','for','what','how','is','of','to','in','a','an','là','gì','và','của','cho','có','về','như','thế','nào','tôi','hãy','được','các','này','đó',
            'mô','hình','trong','những','hiện','tại','đạt','độ','chính','xác','bao','nhiêu','với','từ','đang','để','đã','nó','dự','án','gồm','dùng','sử','dụng','paper','bài','báo'}
    return [w for w in re.findall(r"[^\W]+", text.lower(), re.UNICODE) if len(w)>1 and w not in stop]


class BM25TFIDFIndex:
    """Lightweight, deterministic BM25/TF-IDF inverted index that serializes to JSON."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: List[int] = []
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]

    def fit(self, documents: List[str]):
        self.doc_count = len(documents)
        if self.doc_count == 0:
            return

        total_tokens = 0
        self.doc_lengths = []
        doc_term_freqs: List[Dict[str, int]] = []
        doc_freq: Dict[str, int] = {}

        for doc in documents:
            tokens = tokenize(doc)
            self.doc_lengths.append(len(tokens))
            total_tokens += len(tokens)

            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            doc_term_freqs.append(tf_map)

            for t in tf_map.keys():
                doc_freq[t] = doc_freq.get(t, 0) + 1

        self.avg_doc_len = total_tokens / max(1, self.doc_count)

        # Compute Robertson-Spärck Jones IDF
        self.idf = {}
        for term, df in doc_freq.items():
            # BM25 standard idf: log((N - n + 0.5) / (n + 0.5) + 1)
            self.idf[term] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

        # Build inverted index
        self.inverted_index = {}
        for doc_idx, tf_map in enumerate(doc_term_freqs):
            for term, count in tf_map.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, count))

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.doc_count, dtype=np.float32)
        q_tokens = tokenize(query)
        if not q_tokens or self.doc_count == 0:
            return scores

        for token in q_tokens:
            if token not in self.inverted_index:
                continue
            idf = self.idf.get(token, 0.0)
            postings = self.inverted_index[token]
            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                scores[doc_idx] += idf * (numerator / max(1e-6, denominator))

        return scores

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "idf": self.idf,
            "inverted_index": self.inverted_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BM25TFIDFIndex:
        idx = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        idx.doc_count = data.get("doc_count", 0)
        idx.avg_doc_len = data.get("avg_doc_len", 0.0)
        idx.doc_lengths = data.get("doc_lengths", [])
        idx.idf = data.get("idf", {})
        idx.inverted_index = data.get("inverted_index", {})
        return idx


class Indexer:
    """Manages creation, incremental persistence, and search of document chunks."""

    def __new__(cls, config: Optional[RAGConfig] = None):
        resolved = config or load_rag_config()
        if resolved.knowledge_base.discover_project:
            from src.rag.project_index import ProjectIndexer
            return ProjectIndexer(resolved)
        return super().__new__(cls)

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or load_rag_config()
        self.kb = KnowledgeBase(self.config)
        self.index_dir = self.config.knowledge_base.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.chunks_path = self.index_dir / "chunks.json"
        self.meta_path = self.index_dir / "index_meta.json"
        self.bm25_path = self.index_dir / "bm25_index.json"

        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25TFIDFIndex] = None
        self.file_hashes: Dict[str, str] = {}
        self.knowledge_version = self.config.knowledge_base.version

        # Load existing index if present
        if self.chunks_path.exists() and self.meta_path.exists():
            self._load_from_disk()

    def _load_from_disk(self):
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.file_hashes = meta.get("file_hashes", {})
                self.knowledge_version = meta.get("knowledge_version", self.knowledge_version)

            with open(self.chunks_path, "r", encoding="utf-8") as f:
                raw_chunks = json.load(f)
                self.chunks = [DocumentChunk.from_dict(c) for c in raw_chunks]

            if self.bm25_path.exists():
                with open(self.bm25_path, "r", encoding="utf-8") as f:
                    self.bm25 = BM25TFIDFIndex.from_dict(json.load(f))
            else:
                self._rebuild_bm25()
            logger.info(f"Loaded index with {len(self.chunks)} chunks across {len(self.file_hashes)} files.")
        except Exception as e:
            logger.warning(f"Failed to load cached index, rebuilding: {e}")
            self.rebuild()

    def _rebuild_bm25(self):
        docs = [f"{c.title}\n{c.content}" for c in self.chunks]
        self.bm25 = BM25TFIDFIndex()
        self.bm25.fit(docs)

    def _save_to_disk(self):
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)

        meta = {
            "knowledge_version": self.knowledge_version,
            "indexed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "chunk_count": len(self.chunks),
            "file_count": len(self.file_hashes),
            "file_hashes": self.file_hashes,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if self.bm25:
            with open(self.bm25_path, "w", encoding="utf-8") as f:
                json.dump(self.bm25.to_dict(), f, ensure_ascii=False)

    def compute_file_hash(self, rel_path: str) -> Optional[str]:
        abs_path = self.config.project_root / rel_path
        if not abs_path.exists() or not abs_path.is_file():
            return None
        hasher = hashlib.sha256()
        with open(abs_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def update_index(self) -> Dict[str, Any]:
        """Incrementally update the index: add new, update modified, remove deleted files."""
        current_allowlist = set(self.config.knowledge_base.allowlist_files)
        files_added = []
        files_modified = []
        files_removed = []

        # Detect deleted files
        stored_files = set(self.file_hashes.keys())
        for old_file in stored_files:
            abs_p = self.config.project_root / old_file
            if old_file not in current_allowlist or not abs_p.exists():
                files_removed.append(old_file)

        # Detect new and modified files
        for rel_file in current_allowlist:
            if not self.kb.is_file_allowed(rel_file):
                continue
            cur_hash = self.compute_file_hash(rel_file)
            if cur_hash is None:
                continue

            old_hash = self.file_hashes.get(rel_file)
            if old_hash is None:
                files_added.append(rel_file)
            elif old_hash != cur_hash:
                files_modified.append(rel_file)

        if not files_added and not files_modified and not files_removed and self.chunks:
            logger.info("Index is already up to date. No files modified.")
            return {
                "status": "up_to_date",
                "chunks_count": len(self.chunks),
                "files_count": len(self.file_hashes),
                "knowledge_version": self.knowledge_version,
            }

        logger.info(f"Updating index: +{len(files_added)} added, ~{len(files_modified)} modified, -{len(files_removed)} removed")

        # Filter out chunks of removed or modified files
        keep_files = set(self.file_hashes.keys()) - set(files_removed) - set(files_modified)
        new_chunks = [c for c in self.chunks if c.source_path in keep_files]
        new_hashes = {f: h for f, h in self.file_hashes.items() if f in keep_files}

        # Ingest added and modified files
        for rel_file in files_added + files_modified:
            file_chunks = self.kb.ingest_file(rel_file)
            new_chunks.extend(file_chunks)
            file_hash = self.compute_file_hash(rel_file)
            if file_hash:
                new_hashes[rel_file] = file_hash

        self.chunks = new_chunks
        self.file_hashes = new_hashes
        self._rebuild_bm25()
        self._save_to_disk()

        return {
            "status": "updated",
            "added": files_added,
            "modified": files_modified,
            "removed": files_removed,
            "chunks_count": len(self.chunks),
            "files_count": len(self.file_hashes),
            "knowledge_version": self.knowledge_version,
        }

    def rebuild(self) -> Dict[str, Any]:
        """Completely rebuilds the index from scratch."""
        logger.info("Rebuilding knowledge index from scratch...")
        all_chunks: List[DocumentChunk] = []
        hashes: Dict[str, str] = {}

        for rel_file in self.config.knowledge_base.allowlist_files:
            if not self.kb.is_file_allowed(rel_file):
                continue
            abs_p = self.config.project_root / rel_file
            if not abs_p.exists():
                continue
            chunks = self.kb.ingest_file(rel_file)
            all_chunks.extend(chunks)
            f_hash = self.compute_file_hash(rel_file)
            if f_hash:
                hashes[rel_file] = f_hash

        self.chunks = all_chunks
        self.file_hashes = hashes
        self._rebuild_bm25()
        self._save_to_disk()

        return {
            "status": "rebuilt",
            "chunks_count": len(self.chunks),
            "files_count": len(self.file_hashes),
            "knowledge_version": self.knowledge_version,
        }

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search the index for the query, returning (chunk, score) pairs."""
        if not self.chunks or self.bm25 is None:
            return []

        scores = self.bm25.score(query)
        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.chunks[idx], score))
        return results
