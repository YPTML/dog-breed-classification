"""Document ingestion, AST code chunking, and markdown parsing for RAG knowledge base."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from src.rag.config import RAGConfig, load_rag_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DocumentChunk:
    chunk_id: str
    source_path: str
    title: str
    content: str
    start_line: int
    end_line: int
    doc_type: str  # "markdown", "code", "structured_json", "structured_csv", "config"
    content_hash: str
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentChunk:
        return cls(**data)


def compute_text_hash(text: str) -> str:
    """Computes SHA-256 of text normalized for line endings."""
    normalized = text.replace("\r\n", "\n").strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


class MarkdownParser:
    """Splits markdown by headings while preserving hierarchy and line ranges."""

    HEADING_REGEX = re.compile(r"^(#{1,4})\s+(.+)$")

    @classmethod
    def parse_file(cls, path: Path, rel_path: str, version: str) -> List[DocumentChunk]:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        chunks: List[DocumentChunk] = []
        current_headings: List[Tuple[int, str]] = []  # (level, title)
        current_lines: List[str] = []
        chunk_start_line = 1

        def flush_chunk(end_line: int):
            nonlocal current_lines, chunk_start_line
            text = "".join(current_lines).strip()
            if text:
                title_hierarchy = " > ".join(t for _, t in current_headings) if current_headings else "Document Overview"
                c_hash = compute_text_hash(text)
                chunk_id = f"{rel_path}#L{chunk_start_line}-L{end_line}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    source_path=rel_path,
                    title=title_hierarchy,
                    content=text,
                    start_line=chunk_start_line,
                    end_line=end_line,
                    doc_type="markdown",
                    content_hash=c_hash,
                    version=version,
                    metadata={"headings": [t for _, t in current_headings]},
                ))
            current_lines = []

        for idx, line in enumerate(lines, start=1):
            match = cls.HEADING_REGEX.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()

                # Flush previous content if it exists
                if current_lines:
                    flush_chunk(idx - 1)

                # Update heading stack
                while current_headings and current_headings[-1][0] >= level:
                    current_headings.pop()
                current_headings.append((level, title))
                chunk_start_line = idx

            current_lines.append(line)

        if current_lines:
            flush_chunk(len(lines))

        return chunks


class PythonCodeParser:
    """Parses Python source files using AST to extract module, classes, and functions."""

    @classmethod
    def parse_file(cls, path: Path, rel_path: str, version: str) -> List[DocumentChunk]:
        with open(path, "r", encoding="utf-8") as f:
            source_text = f.read()

        lines = source_text.splitlines(keepends=True)
        chunks: List[DocumentChunk] = []

        try:
            tree = ast.parse(source_text, filename=str(path))
        except SyntaxError:
            # Fallback to simple line-based chunk if syntax error
            c_hash = compute_text_hash(source_text)
            return [DocumentChunk(
                chunk_id=f"{rel_path}#L1-L{len(lines)}",
                source_path=rel_path,
                title=f"{rel_path} (Full Module)",
                content=source_text[:3000],
                start_line=1,
                end_line=len(lines),
                doc_type="code",
                content_hash=c_hash,
                version=version,
                metadata={"type": "module"},
            )]

        # 1. Module-level docstring and header overview
        mod_docstring = ast.get_docstring(tree) or ""
        header_end = 1
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                header_end = max(1, node.lineno - 1)
                break
        else:
            header_end = len(lines)

        header_lines = lines[:header_end]
        header_text = "".join(header_lines).strip()
        if header_text:
            chunks.append(DocumentChunk(
                chunk_id=f"{rel_path}#L1-L{header_end}",
                source_path=rel_path,
                title=f"{rel_path} > Module Header & Docstring",
                content=header_text,
                start_line=1,
                end_line=header_end,
                doc_type="code",
                content_hash=compute_text_hash(header_text),
                version=version,
                metadata={"type": "module_header", "docstring": mod_docstring},
            ))

        # 2. Extract Top-level Classes and Functions
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                s_line = node.lineno
                e_line = getattr(node, "end_lineno", s_line + 10)
                node_lines = lines[s_line - 1:e_line]
                code_text = "".join(node_lines).strip()
                doc = ast.get_docstring(node) or ""
                chunks.append(DocumentChunk(
                    chunk_id=f"{rel_path}#L{s_line}-L{e_line}",
                    source_path=rel_path,
                    title=f"{rel_path} > func {node.name}()",
                    content=code_text,
                    start_line=s_line,
                    end_line=e_line,
                    doc_type="code",
                    content_hash=compute_text_hash(code_text),
                    version=version,
                    metadata={"type": "function", "name": node.name, "docstring": doc},
                ))
            elif isinstance(node, ast.ClassDef):
                s_line = node.lineno
                e_line = getattr(node, "end_lineno", s_line + 20)
                node_lines = lines[s_line - 1:e_line]
                code_text = "".join(node_lines).strip()
                doc = ast.get_docstring(node) or ""
                chunks.append(DocumentChunk(
                    chunk_id=f"{rel_path}#L{s_line}-L{e_line}",
                    source_path=rel_path,
                    title=f"{rel_path} > class {node.name}",
                    content=code_text,
                    start_line=s_line,
                    end_line=e_line,
                    doc_type="code",
                    content_hash=compute_text_hash(code_text),
                    version=version,
                    metadata={"type": "class", "name": node.name, "docstring": doc},
                ))

        return chunks


class StructuredFileParser:
    """Extracts semantic, human-readable chunks from JSON and CSV files."""

    @classmethod
    def parse_file(cls, path: Path, rel_path: str, version: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        suffix = path.suffix.lower()

        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Split large JSON into logical top-level sections
            if isinstance(data, dict):
                for k, v in data.items():
                    chunk_content = f"JSON File: {rel_path}\nSection: {k}\n\n{json.dumps({k: v}, indent=2, ensure_ascii=False)}"
                    # Truncate overly long dumps
                    if len(chunk_content) > 3500:
                        chunk_content = chunk_content[:3500] + "\n... [truncated for retrieval context]"
                    chunks.append(DocumentChunk(
                        chunk_id=f"{rel_path}#{k}",
                        source_path=rel_path,
                        title=f"{rel_path} > {k}",
                        content=chunk_content,
                        start_line=1,
                        end_line=1,
                        doc_type="structured_json",
                        content_hash=compute_text_hash(chunk_content),
                        version=version,
                        metadata={"section": k},
                    ))
            else:
                text = json.dumps(data, indent=2, ensure_ascii=False)[:3500]
                chunks.append(DocumentChunk(
                    chunk_id=f"{rel_path}#root",
                    source_path=rel_path,
                    title=f"{rel_path} > Overview",
                    content=f"JSON File: {rel_path}\n\n{text}",
                    start_line=1,
                    end_line=1,
                    doc_type="structured_json",
                    content_hash=compute_text_hash(text),
                    version=version,
                    metadata={},
                ))

        elif suffix == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rows = list(reader)

            header_str = ", ".join(header) if header else ""
            summary_lines = [
                f"CSV File: {rel_path}",
                f"Total Rows: {len(rows)}",
                f"Columns: {header_str}",
                "\nSample Rows (First 15):",
            ]
            for r in rows[:15]:
                summary_lines.append(" | ".join(r))

            text = "\n".join(summary_lines)
            chunks.append(DocumentChunk(
                chunk_id=f"{rel_path}#summary",
                source_path=rel_path,
                title=f"{rel_path} > Table Summary",
                content=text,
                start_line=1,
                end_line=min(20, len(rows) + 1),
                doc_type="structured_csv",
                content_hash=compute_text_hash(text),
                version=version,
                metadata={"total_rows": len(rows), "header": header},
            ))

        elif suffix in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            chunks.append(DocumentChunk(
                chunk_id=f"{rel_path}#config",
                source_path=rel_path,
                title=f"{rel_path} > Configuration",
                content=f"YAML Configuration: {rel_path}\n\n{content}",
                start_line=1,
                end_line=len(content.splitlines()),
                doc_type="config",
                content_hash=compute_text_hash(content),
                version=version,
                metadata={},
            ))

        elif suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            chunks.append(DocumentChunk(
                chunk_id=f"{rel_path}#text",
                source_path=rel_path,
                title=f"{rel_path} > Text Info",
                content=content,
                start_line=1,
                end_line=len(content.splitlines()),
                doc_type="markdown",
                content_hash=compute_text_hash(content),
                version=version,
                metadata={},
            ))

        return chunks


class KnowledgeBase:
    """Manages document ingestion, chunking, and file allowlisting."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or load_rag_config()
        self.root = self.config.project_root

    def is_file_allowed(self, rel_path: str) -> bool:
        """Strict check against allowlist and exclusion patterns."""
        if self.config.knowledge_base.discover_project:
            from src.rag.project_documents import permitted
            return permitted(self.config, rel_path, metadata=True)
        normalized = rel_path.replace("\\", "/")

        # Check explicit exclusions
        for pattern in self.config.knowledge_base.exclude_patterns:
            clean_pat = pattern.replace("**", "").replace("*", "")
            if clean_pat in normalized or normalized.endswith(pattern.replace("*", "")):
                return False

        # Check allowlist
        for allowed in self.config.knowledge_base.allowlist_files:
            allowed_norm = allowed.replace("\\", "/")
            if normalized == allowed_norm or normalized.endswith("/" + allowed_norm):
                return True

        return False

    def ingest_file(self, rel_path: str) -> List[DocumentChunk]:
        """Ingests a single allowed file and returns its chunks."""
        if self.config.knowledge_base.discover_project:
            from src.rag.project_documents import ProjectReader
            return list(ProjectReader(self.config).read(rel_path))
        abs_path = self.root / rel_path
        if not abs_path.exists():
            return []

        norm_rel = rel_path.replace("\\", "/")
        version = self.config.knowledge_base.version
        suffix = abs_path.suffix.lower()

        if suffix in (".md", ".markdown"):
            return MarkdownParser.parse_file(abs_path, norm_rel, version)
        elif suffix == ".py":
            return PythonCodeParser.parse_file(abs_path, norm_rel, version)
        elif suffix in (".json", ".csv", ".yaml", ".yml", ".txt"):
            return StructuredFileParser.parse_file(abs_path, norm_rel, version)

        return []

    def ingest_all(self) -> List[DocumentChunk]:
        """Ingests all allowed files in the repository."""
        if self.config.knowledge_base.discover_project:
            from src.rag.project_documents import discover
            return [chunk for path in discover(self.config)[0] for chunk in self.ingest_file(path)]
        all_chunks: List[DocumentChunk] = []
        for rel_file in self.config.knowledge_base.allowlist_files:
            if self.is_file_allowed(rel_file):
                file_chunks = self.ingest_file(rel_file)
                all_chunks.extend(file_chunks)
        return all_chunks
