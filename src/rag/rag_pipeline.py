"""Main RAG Pipeline orchestrating hybrid retrieval, evidence gating, and generation."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.rag.config import RAGConfig, load_rag_config
from src.rag.indexer import Indexer
from src.rag.llm_client import LLMClient, LLMResponse
from src.rag.retriever import HybridRetriever, RetrievalResult
from src.rag.structured_engine import StructuredEngine

logger = logging.getLogger("rag_pipeline")

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|system)\s+instructions?",
    r"bỏ\s+qua\s+(hết\s+)?(các\s+)?(chỉ\s+dẫn|hướng\s+dẫn|lệnh)",
    r"từ\s+bây\s+giờ\s+hãy\s+đóng\s+vai",
    r"you\s+are\s+now",
    r"system\s*prompt\s*injection",
    r"hãy\s+nói\s+rằng\s+mô\s+hình\s+đạt\s+100%",
    r"hãy\s+khẳng\s+định\s+chính\s+xác\s+tuyệt\s+đối",
]

OUT_OF_SCOPE_TOPICS = [
    r"chăm\s+sóc\s+(chó|thú\s+cưng)",
    r"cho\s+ăn\s+gì",
    r"thức\s+ăn\s+cho",
    r"dinh\s+dưỡng",
    r"chữa\s+bệnh",
    r"tiêm\s+phòng",
    r"mua\s+bán\s+chó",
    r"giá\s+(tiền\s+)?(mua\s+)?(chó|husky|poodle|corgi)",
    r"phối\s+giống",
]


class RAGPipeline:
    """End-to-end RAG orchestrator for Dog Breed Recognition."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or load_rag_config()
        self.indexer = Indexer(self.config)
        self.retriever = HybridRetriever(indexer=self.indexer, config=self.config)
        self.llm_client = LLMClient(self.config.llm)
        self.structured_engine = self.retriever.structured_engine
        self.knowledge_version = self.indexer.knowledge_version

    def check_prompt_injection(self, text: str) -> bool:
        """Detect attempts to override system guidelines or inject instructions."""
        t_low = text.lower()
        for pat in INJECTION_PATTERNS:
            if re.search(pat, t_low):
                return True
        return False

    def check_out_of_scope(self, text: str) -> bool:
        """Detect questions regarding general pet care, veterinary advice, or puppy pricing."""
        t_low = text.lower()
        for pat in OUT_OF_SCOPE_TOPICS:
            if re.search(pat, t_low):
                return True
        return False

    def query(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        last_prediction: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Processes user question and returns grounded answer with full citations and metadata."""
        clean_q = question.strip()
        if hasattr(self.indexer, 'reload'):
            self.indexer.reload()
            if self.knowledge_version != self.indexer.knowledge_version:
                self.structured_engine = StructuredEngine(self.config.project_root)
                self.retriever.structured_engine = self.structured_engine
            self.knowledge_version = self.indexer.knowledge_version
        if not clean_q:
            return {
                "answer": "Vui lòng nhập câu hỏi cần tra cứu.",
                "sources": [],
                "knowledge_version": self.knowledge_version,
                "missing_evidence": True,
                "evidence_note": "Câu hỏi rỗng.",
                "llm_status": "none",
                "llm_provider": self.config.llm.provider,
            }

        # 1. Prompt Injection Defense
        if self.check_prompt_injection(clean_q):
            logger.warning(f"Adversarial prompt injection pattern detected: {clean_q}")
            return {
                "answer": (
                    "**Yêu cầu không được chấp nhận:** Hệ thống hoạt động theo chính sách bảo mật dữ liệu nghiêm ngặt. "
                    "Toàn bộ tài liệu tham khảo và câu hỏi của người dùng chỉ được xử lý như dữ liệu thuần túy, không được "
                    "thay đổi vai trò trợ lý hay làm sai lệch các số liệu thực nghiệm đã kiểm chứng."
                ),
                "sources": [{"source_path": "configs/rag_config.yaml", "title": "Security & Alignment Rules"}],
                "knowledge_version": self.knowledge_version,
                "missing_evidence": True,
                "evidence_note": "Prompt injection pattern detected and neutralized.",
                "llm_status": "neutralized",
                "llm_provider": self.config.llm.provider,
            }

        # 3. Hybrid Retrieval
        retrieval_res: RetrievalResult = self.retriever.retrieve(
            query=clean_q,
            history=history,
            top_k=self.config.retrieval.top_k,
        )

        # 4. Check evidence presence
        has_chunks = bool(retrieval_res.chunks)
        has_structured = bool(retrieval_res.structured_data)
        has_pred = bool(last_prediction)

        if self.check_out_of_scope(clean_q):
            care_terms = ('chăm sóc','dinh dưỡng','thức ăn','tiêm phòng','chữa bệnh','veterinary','nutrition','feeding')
            breed_words = {word for breed in self.structured_engine.get_class_names().values()
                           for word in breed.lower().replace('_',' ').split() if len(word)>2}
            requested_breed_words = breed_words.intersection(re.findall(r'[^\W]+',clean_q.lower()))
            relevant = [c for c in retrieval_res.chunks if any(t in c.content.lower() for t in care_terms)
                        and all(t in c.content.lower() for t in requested_breed_words)
                        and not c.source_path.startswith(('tests/','src/rag/'))]
            if not relevant:
                return {'answer':'Kho kiến thức không có dữ liệu phù hợp để trả lời câu hỏi này.',
                        'sources':[],'knowledge_version':self.knowledge_version,'missing_evidence':True,
                        'evidence_note':'Chưa có tài liệu liên quan được truy xuất.', 'llm_status':'no_evidence',
                        'llm_provider':self.config.llm.provider}
            retrieval_res.chunks = relevant
            retrieval_res.structured_data = None
            relevant_ids = {c.chunk_id for c in relevant}
            retrieval_res.sources = [s for s in retrieval_res.sources if s.get('chunk_id') in relevant_ids]

        if not has_chunks and not has_structured and not has_pred:
            return {
                "answer": (
                    "**Kho kiến thức không có bằng chứng:** Hiện tại không tìm thấy thông tin hoặc số liệu liên quan "
                    "đến câu hỏi trong tài liệu, mã nguồn và kết quả thực nghiệm đã lưu của dự án. "
                    "Hệ thống không suy đoán khi thiếu căn cứ đối chiếu."
                ),
                "sources": [],
                "knowledge_version": self.knowledge_version,
                "missing_evidence": True,
                "evidence_note": "Không tìm thấy đoạn tài liệu hoặc số liệu phù hợp trong kho kiến thức.",
                "llm_status": "no_evidence",
                "llm_provider": self.config.llm.provider,
            }

        # 5. Generate Answer via LLM Client
        chunk_dicts = [c.to_dict() for c in retrieval_res.chunks]
        llm_res: LLMResponse = self.llm_client.generate(
            query=clean_q,
            context_chunks=chunk_dicts,
            structured_info=retrieval_res.structured_data,
            history=history,
            prediction_context=last_prediction,
        )

        # Compile final sources list
        sources_list = []
        if retrieval_res.structured_data and "source_file" in retrieval_res.structured_data:
            sources_list.append({
                "source_path": retrieval_res.structured_data["source_file"],
                "title": f"Dữ liệu tính toán xác định: {retrieval_res.structured_data.get('type')}",
                "relevance_score": 1.0,
            })
        for s in retrieval_res.sources:
            if not any(x.get("source_path") == s.get("source_path") and
                       x.get("chunk_id") == s.get("chunk_id") and x.get("locator") == s.get("locator") for x in sources_list):
                sources_list.append(s)

        return {
            "answer": llm_res.answer,
            "sources": sources_list,
            "knowledge_version": self.knowledge_version,
            "missing_evidence": retrieval_res.structured_data is not None and retrieval_res.structured_data.get('type')=='paper_inventory' and not retrieval_res.structured_data.get('papers'),
            "evidence_note": 'Chưa có paper PDF trong chỉ mục.' if retrieval_res.structured_data and retrieval_res.structured_data.get('type')=='paper_inventory' and not retrieval_res.structured_data.get('papers') else None,
            "llm_status": llm_res.llm_status,
            "llm_provider": llm_res.provider,
            "llm_model": llm_res.model,
            "missing_config_message": llm_res.missing_config_message,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return system status, knowledge version, chunk/file count, and LLM readiness."""
        is_live, missing_msg = self.llm_client.is_live_available()
        if hasattr(self.indexer,'reload'):
            self.indexer.reload()
        coverage = getattr(self.indexer,'coverage',{})
        return {
            "status": "ok" if len(self.indexer.chunks) else "index_missing",
            "knowledge_version": self.indexer.knowledge_version,
            "chunks_count": len(self.indexer.chunks),
            "files_count": coverage.get('indexed_files', len(self.indexer.file_hashes)),
            "llm_status": "live" if is_live else "missing_config",
            "llm_provider": self.config.llm.provider,
            "llm_model": self.config.llm.model,
            "missing_config_message": missing_msg,
            "paper_files": coverage.get('paper_files',0),
            "index_issues": coverage.get('issues_count',0),
            "index_dir": str(self.config.knowledge_base.index_dir),
            "coverage": {k:v for k,v in coverage.items() if k in ('scanned_files','by_extension','indexed_files','complete_files','image_note','indexed_at_utc')},
            "retrieval_backend": "SQLite FTS5 + structured data" if hasattr(self.indexer,'db_path') else 'BM25 + structured data',
        }



# Global singleton instance for the app
_pipeline_instance: Optional[RAGPipeline] = None


def get_rag_pipeline(config: Optional[RAGConfig] = None) -> RAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline(config)
    return _pipeline_instance
