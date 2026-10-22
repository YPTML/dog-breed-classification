"""Configuration loader and schema definitions for RAG system."""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class KnowledgeBaseConfig:
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "rag_index")
    version: str = "2026-10-17-v1"
    allowlist_files: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    discover_project: bool = False
    max_file_bytes: int = 52428800


@dataclass
class ChunkingConfig:
    markdown_max_tokens: int = 500
    code_max_tokens: int = 400
    overlap_tokens: int = 50


@dataclass
class RetrievalConfig:
    top_k: int = 4
    min_relevance_score: float = 0.15
    dense_weight: float = 0.5
    lexical_weight: float = 0.5
    local_query_rewrite: bool = True


@dataclass
class LLMConfig:
    provider: str = "auto"  # "gemini", "openrouter", "openai", "offline_grounded"
    model: str = "gemini-2.5-flash"
    openrouter_model: str = "google/gemini-2.5-flash"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_seconds: int = 20
    api_key: Optional[str] = None
    api_base: Optional[str] = None


@dataclass
class RAGConfig:
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def load_rag_config(config_path: Optional[Path] = None) -> RAGConfig:
    """Load RAG configuration from YAML file and apply environment variable overrides."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "rag_config.yaml"

    raw_cfg: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

    kb_raw = raw_cfg.get("knowledge_base", {})
    index_dir = os.environ.get('RAG_INDEX_DIR') or kb_raw.get("index_dir", "outputs/rag_index")
    index_path = Path(index_dir)
    if not index_path.is_absolute():
        index_path = PROJECT_ROOT / index_path
    if not os.environ.get('RAG_INDEX_DIR') and kb_raw.get('use_local_appdata') and os.environ.get('LOCALAPPDATA'):
        project_key = hashlib.sha256(str(PROJECT_ROOT.resolve()).casefold().encode()).hexdigest()[:12]
        index_path = Path(os.environ['LOCALAPPDATA']) / 'DogBreedRAG' / 'indexes' / project_key

    kb_config = KnowledgeBaseConfig(
        index_dir=index_path,
        version=str(kb_raw.get("version", "2026-10-17-v1")),
        allowlist_files=list(kb_raw.get("allowlist_files", [])),
        exclude_patterns=list(kb_raw.get("exclude_patterns", [])),
        discover_project=bool(kb_raw.get("discover_project", False)),
        max_file_bytes=int(kb_raw.get("max_file_bytes", 52428800)),
    )

    chunk_raw = raw_cfg.get("chunking", {})
    chunk_config = ChunkingConfig(
        markdown_max_tokens=int(chunk_raw.get("markdown_max_tokens", 500)),
        code_max_tokens=int(chunk_raw.get("code_max_tokens", 400)),
        overlap_tokens=int(chunk_raw.get("overlap_tokens", 50)),
    )

    retrieval_raw = raw_cfg.get("retrieval", {})
    retrieval_config = RetrievalConfig(
        top_k=int(retrieval_raw.get("top_k", 4)),
        min_relevance_score=float(retrieval_raw.get("min_relevance_score", 0.15)),
        dense_weight=float(retrieval_raw.get("dense_weight", 0.5)),
        lexical_weight=float(retrieval_raw.get("lexical_weight", 0.5)),
        local_query_rewrite=bool(retrieval_raw.get("local_query_rewrite", True)),
    )

    llm_raw = raw_cfg.get("llm", {})
    env_provider = os.environ.get("RAG_LLM_PROVIDER")
    provider = env_provider or str(llm_raw.get("provider", "auto"))

    # Resolve API keys from environment
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Clean placeholder keys
    if openrouter_key and "YOUR_KEY_HERE" in openrouter_key:
        openrouter_key = None
    if gemini_key and "YOUR_KEY_HERE" in gemini_key:
        gemini_key = None
    if openai_key and "YOUR_KEY_HERE" in openai_key:
        openai_key = None

    resolved_key = None
    resolved_provider = provider

    if provider == "auto":
        if gemini_key:
            resolved_provider = "gemini"
            resolved_key = gemini_key
        elif openrouter_key:
            resolved_provider = "openrouter"
            resolved_key = openrouter_key
        elif openai_key:
            resolved_provider = "openai"
            resolved_key = openai_key
        elif os.environ.get('OLLAMA_HOST'):
            resolved_provider = 'ollama'
        else:
            resolved_provider = "offline_grounded"
    elif provider == "gemini":
        resolved_key = gemini_key
    elif provider == "openrouter":
        resolved_key = openrouter_key
    elif provider == "openai":
        resolved_key = openai_key

    llm_config = LLMConfig(
        provider=resolved_provider,
        model=os.environ.get("RAG_LLM_MODEL") or (str(llm_raw.get('ollama_model','qwen2.5:3b')) if resolved_provider=='ollama' else str(llm_raw.get("model", "gemini-2.5-flash"))),
        openrouter_model=os.environ.get("RAG_LLM_MODEL") or str(llm_raw.get("openrouter_model", "google/gemini-2.5-flash")),
        temperature=float(llm_raw.get("temperature", 0.1)),
        max_tokens=int(llm_raw.get("max_tokens", 1024)),
        timeout_seconds=int(llm_raw.get("timeout_seconds", 20)),
        api_key=resolved_key,
        api_base=(os.environ.get('OLLAMA_HOST') or llm_raw.get('ollama_base') or 'http://127.0.0.1:11434') if resolved_provider=='ollama' else os.environ.get("OPENAI_BASE_URL") or llm_raw.get("api_base"),
    )

    return RAGConfig(
        project_root=PROJECT_ROOT,
        knowledge_base=kb_config,
        chunking=chunk_config,
        retrieval=retrieval_config,
        llm=llm_config,
    )
