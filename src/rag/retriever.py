"""Hybrid Retriever with Vietnamese-English entity resolution and structured routing."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.rag.config import RAGConfig, load_rag_config
from src.rag.indexer import Indexer
from src.rag.knowledge_base import DocumentChunk
from src.rag.structured_engine import StructuredEngine
from src.rag.query_rewriter import LocalQueryRewriter

# Synonym and translation mappings: Vietnamese user phrasing -> Canonical English / technical terms
TERM_MAPPINGS: Dict[str, List[str]] = {
 'tóm tắt': ['abstract', 'introduction', 'conclusion'],
 'đặc trưng': ['features', 'descriptors', 'representation'],
 'phương pháp': ['method', 'approach', 'pipeline'],
 'thuật toán': ['algorithm', 'classifier'],
 'tác giả': ['author'],
 'kết quả': ['results', 'evaluation'],
'mục đích': ['purpose', 'overview', 'classification', 'application'],
 'dùng để': ['purpose', 'overview', 'classification'],
 'dùng cho': ['purpose', 'overview', 'classification'],
 'huấn luyện': ['training', 'train', 'epochs', 'optimizer'],
 'học chuyển giao': ['transfer', 'learning', 'pretrained', 'backbone'],
 'độ chính xác': ['accuracy', 'validation', 'results'],
 'đóng góp': ['contribution', 'proposed', 'method'],
 'tích chập': ['convolution', 'depthwise', 'pointwise'],
 'khối dư': ['residual', 'inverted', 'bottleneck'],
 'husky': ['siberian_husky', 'malamute', 'eskimo_dog'],
 'siberian husky': ['siberian_husky'],
 'shiba': ['shiba_inu', 'unsupported_dog_breed', 'oxford-iiit'],
 'shiba inu': ['shiba_inu', 'unsupported_dog_breed', 'oxford-iiit'],
 'phú quốc': ['phu_quoc_ridgeback', 'wikimedia', 'audit'],
 'phu quoc': ['phu_quoc_ridgeback', 'wikimedia', 'audit'],
 'chó phú quốc': ['phu_quoc_ridgeback', 'wikimedia', 'audit'],
 'poodle': ['miniature_poodle', 'toy_poodle', 'standard_poodle'],
 'schnauzer': ['standard_schnauzer', 'miniature_schnauzer', 'giant_schnauzer'],
 'collie': ['collie', 'border_collie', 'shetland_sheepdog'],
 'chó tuyết': ['siberian_husky', 'malamute', 'eskimo_dog', 'samoyed'],
 'chó chăn cừu': ['collie', 'border_collie', 'shetland_sheepdog', 'german_shepherd'],
 'garfield': ['garfield', 'verified_non_dog_garfield'],
 'snoopy': ['snoopy', 'cartoon_dog_snoopy'],
 'cougar': ['cougar', 'animals_non_dog/cougar_body', 'false_dog'],
 'kangaroo': ['kangaroo', 'false_dog'],
 'okapi': ['okapi', 'false_dog'],
 'f1 thấp nhất': ['lowest_f1', 'per_class_metrics'],
 'f1 cao nhất': ['per_class_metrics'],
 'bao nhiêu giống': ['classes', 'num_classes', 'class_names.json'],
 'hỗ trợ bao nhiêu giống': ['classes', 'num_classes', 'class_names.json'],
 'bao nhiêu ảnh': ['sample_count', 'support'],
 'nhận sai': ['misclassified', 'error_examples', 'top_confusions'],
 'dự đoán sai': ['misclassified', 'error_examples', 'top_confusions'],
 'top-1': ['top1', 'accuracy'],
 'top-5': ['top5', 'accuracy'],
 'top 1': ['top1', 'accuracy'],
 'top 5': ['top5', 'accuracy'],
 'ngưỡng ood': ['T_uncertain', 'T_not_dog', 'balanced', 'ood_calibration.json'],
 'ngưỡng': ['T_uncertain', 'T_not_dog'],
 'độ trễ': ['latency', 'median', 'ms'],
 'độ ổn định': ['robustness', 'webp_q80', 'jpeg_q35', 'resize_96_jpeg_q85'],
 'ảnh nhỏ': ['resize_96_jpeg_q85', 'robustness'],
 'thu nhỏ': ['resize_96_jpeg_q85', 'robustness'],
 'dog': ['DOG', 'UNCERTAIN', 'NOT_DOG'],
 'uncertain': ['UNCERTAIN'],
 'not_dog': ['NOT_DOG'],
 'tiền xử lý': ['preprocessing'],
 'kích thước đầu vào': ['input_shape'],
 'kiến trúc': ['mobilenetv2', 'architecture', 'backbone', 'dense', 'EXPECTED_INPUT_SHAPE', 'EXPECTED_EMBEDDING_DIM', 'embedding_dimension', 'class_count'],
 'centroid': ['class_centroids.npy', 'cosine'],
 'lỗi đã sửa': ['bugfix', 'BUGFIX_REPORT.md'],
 'kiểm thử': ['test', 'unittest', 'regression']}

TERM_MAPPINGS['trùng'] = ['duplicate', 'duplicates', 'near_duplicate', 'leakage', 'overlap']


@dataclass
class RetrievalResult:
    query: str
    expanded_query: str
    chunks: List[DocumentChunk]
    structured_data: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)


class HybridRetriever:
    """Combines BM25 lexical retrieval, entity synonym expansion, and structured query routing."""

    def __init__(self, indexer: Optional[Indexer] = None, config: Optional[RAGConfig] = None):
        self.config = config or load_rag_config()
        self.indexer = indexer or Indexer(self.config)
        self.structured_engine = StructuredEngine(self.config.project_root)
        self.query_rewriter = LocalQueryRewriter(self.config)

    def _search(self, query, alternate='', *, top_k=5, **scope):
        original = self.indexer.search(query, top_k=top_k, **scope)
        if not alternate or alternate == query:
            return original
        translated = self.indexer.search(alternate, top_k=top_k, **scope)
        # Rank fusion retains Vietnamese evidence and adds English passages.
        ranked = {}
        for results, weight in ((original, 1.0), (translated, 1.0)):
            for rank, (chunk, _) in enumerate(results, 1):
                prior = ranked.get(chunk.chunk_id, (chunk, 0.0))[1]
                ranked[chunk.chunk_id] = (chunk, prior + weight / (60 + rank))
        fused = sorted(ranked.values(), key=lambda pair: pair[1], reverse=True)
        # A translation can omit useful Vietnamese notes. Preserve the strongest
        # hit of each query before filling the remaining slots with fused ranks.
        anchors = [ranked[results[0][0].chunk_id] for results in (original, translated) if results]
        candidates = anchors + fused if top_k > 1 else fused
        unique = {}
        for chunk, score in candidates:
            unique.setdefault(chunk.chunk_id, (chunk, score))
        return list(unique.values())[:top_k]

    def expand_query(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Expands Vietnamese and short queries with canonical terms and conversation context."""
        expanded_tokens = [query]
        q_lower = query.lower()

        # Check for pronouns requiring conversation resolution
        pronouns = ["nó", "giống đó", "giống này", "còn giống", "thế còn", "vậy còn"]
        if history and any(p in q_lower for p in pronouns):
            # Scan last assistant and user turns for recognized entities
            for turn in reversed(history[-4:]):
                content = turn.get("content", "").lower()
                for key in TERM_MAPPINGS.keys():
                    if key in content:
                        expanded_tokens.append(key)
                        break

        # Match synonyms and terms
        for phrase, canonical_terms in TERM_MAPPINGS.items():
            if phrase in q_lower:
                expanded_tokens.extend(canonical_terms)

        return " ".join(expanded_tokens)

    def route_structured_query(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
        """Detects analytical/factual questions and resolves them directly via StructuredEngine."""
        q = query.lower()

        if not any(t in q for t in ('paper','bài báo')) and any(term in q for term in ('mục đích', 'dùng để làm gì', 'dùng cho', 'chức năng chính')):
            overview = self.config.project_root / 'README.md'
            if overview.exists():
                text = overview.read_text(encoding='utf-8')
                if 'image classifier' in text.lower() and 'uploaded image' in text.lower():
                    names = self.structured_engine.get_class_names()
                    return {'type':'project_purpose', 'explanation':
                            f'Chương trình dùng để nhận diện giống chó từ ảnh người dùng tải lên. '
                            f'Mô hình hiện hỗ trợ {len(names)} giống chó, trả về giống dự đoán và các ứng viên Top-5. '
                            'Bộ kiểm tra OOD đánh giá ảnh trước khi trả trạng thái DOG, UNCERTAIN hoặc NOT_DOG. '
                            'Kết quả là gợi ý phân loại ảnh; trạng thái DOG không bảo đảm tên giống luôn đúng.',
                            'source_file':'README.md'}

        inventory_question = any(t in q for t in ('danh sách paper','danh sách bài báo','đã đọc','đã học','có những paper','có những bài báo')) or bool(
            re.search(r'(bao nhiêu|số lượng)\s+(paper|bài báo)|(paper|bài báo)\s+(nào|đã có)', q))
        if any(t in q for t in ('paper','bài báo')) and (not getattr(self.indexer,'coverage',{}).get('paper_files') or inventory_question):
            papers = [f for f in getattr(self.indexer,'coverage',{}).get('files',[])
                      if f['source_path'].lower().endswith('.pdf') and f['chunks']]
            return {'type':'paper_inventory','explanation':
                    ('Các paper PDF đã lập chỉ mục:\n'+'\n'.join('- '+f['source_path'] for f in papers)) if papers else
                    'Kho chỉ mục hiện chưa có paper PDF đọc được. Hãy đặt paper vào thư mục papers/ và chạy cập nhật chỉ mục. '
                    'PDF dạng ảnh cần OCR; hệ thống sẽ báo trang chưa đọc được thay vì coi là đã học.',
                    'source_file':'outputs/rag_index/coverage.json','papers':papers}

        # A paper's experimental metrics must not be replaced with this project's metrics.
        if any(t in q for t in ('paper','bài báo')):
            return None

        # 1. Supported classes count
        if ("bao nhiêu giống" in q or "mấy giống" in q or "danh sách giống" in q or "số giống" in q) and "hỗ trợ" in q:
            names = self.structured_engine.get_class_names()
            return {
                "type": "class_count",
                "count": len(names),
                "model_version": "oe_v2",
                "sample_classes": [names[i] for i in sorted(names.keys())[:10]],
                "explanation": f"Mô hình OE V2 hiện tại hỗ trợ {len(names)} giống chó theo ánh xạ lớp đang triển khai.",
                "source_file": "outputs/deployment_oe_v2/class_names.json",
            }

        # 2. DOG state certainty vs breed accuracy (read measured data).
        if "dog" in q and any(t in q for t in ("chắc chắn đúng","có nghĩa là","đúng không")):
            m = self.structured_engine.get_global_metrics()
            accepted, correct = m["id_dog_count"], m["id_dog_correct_count"]
            return {"type":"dog_state_meaning","explanation":
                    f"DOG là trạng thái chấp nhận của bộ kiểm tra OOD, không bảo đảm tên giống đúng. "
                    f"Trong lần đánh giá {m['evaluation_date_utc']}, accuracy trên nhóm DOG là "
                    f"{m['accuracy_on_dog_subset']*100:.2f}% ({correct}/{accepted} ảnh); "
                    f"còn {accepted-correct} ảnh đoán sai giống ({(accepted-correct)/accepted*100:.2f}%). Confidence softmax chưa được chứng minh là xác suất đúng đã hiệu chuẩn.",
                    "source_file":m["source_ref"]}

        # 3. Lowest F1 breed
        if ("f1 thấp nhất" in q or "f1 kém nhất" in q or "giống yếu nhất" in q or "f1 nhỏ nhất" in q):
            lowest = self.structured_engine.get_lowest_f1_breeds(10)
            return {
                "type": "lowest_f1_breeds",
                "lowest_breed": lowest[0],
                "top10_weakest": lowest,
                "explanation": (
                    f"Giống có F1 thấp nhất là '{lowest[0]['breed']}' với F1 = {lowest[0]['f1']:.3f} "
                    f"(Precision = {lowest[0]['precision']:.3f}, Recall = {lowest[0]['recall']:.3f}) "
                    f"và có {lowest[0]['support']} ảnh trong tập test."
                ),
                "source_file": "outputs/quality_evaluation_2026-10-16/per_class_metrics.csv",
            }

        # Numerical responses are formatted from the current evaluation artifact.
        if ("top-1" in q or "top 1" in q) and any(t in q for t in ("top-5","top 5","khác","so với")):
            m = self.structured_engine.get_global_metrics()
            return {"type":"top1_vs_top5","explanation":
                    f"Top-1 kiểm tra giống đứng đầu có đúng nhãn thật: {m['top1_accuracy']*100:.2f}% "
                    f"({m['top1_correct_count']}/{m['test_images_count']}), sai {m['top1_misclassified_count']} ảnh. "
                    f"Top-5 kiểm tra nhãn thật có nằm trong 5 ứng viên: {m['top5_accuracy']*100:.2f}% "
                    f"({m['top5_correct_count']}/{m['test_images_count']}), sai {m['top5_misclassified_count']} ảnh. "
                    f"Thời điểm đánh giá: {m['evaluation_date_utc']}.", "source_file":m["source_ref"]}
        if "bao nhiêu ảnh" in q and any(t in q for t in ("nhận sai","dự đoán sai","bị sai","lỗi")):
            m = self.structured_engine.get_global_metrics()
            return {"type":"misclassified_counts","explanation":
                    f"Trong {m['test_images_count']} ảnh chó test: {m['top1_misclassified_count']} ảnh sai Top-1; "
                    f"{m['top5_misclassified_count']} ảnh sai Top-5. Có {m['non_dog_false_dog_count']}/"
                    f"{m['non_dog_test_count']} ảnh không phải chó bị nhận DOG.",
                    "source_file":m["source_ref"]}

        # 5. OOD Thresholds provenance
        if 'ngưỡng' in q and any(t in q for t in ('ood', 'uncertain', 'not_dog', 'không phải chó', 'từ chối')):
            thresholds = self.structured_engine.get_ood_thresholds()
            return {
                "type": "ood_thresholds",
                "thresholds": thresholds,
                "explanation": (
                    f"Ngưỡng OOD hiện tại lấy từ quá trình hiệu chuẩn tại điểm vận hành BALANCED trên tập validation "
                    f"(outputs/deployment_oe_v2/ood_calibration.json), gồm T_uncertain = {thresholds['T_uncertain']} "
                    f"và T_not_dog = {thresholds['T_not_dog']}. Khoảng cách tính bằng 1 - max(cosine similarity) "
                    f"giữa vector embedding 1280 chiều với 120 centroid lớp."
                ),
                "source_file": "outputs/deployment_oe_v2/ood_calibration.json",
            }

        # 6. Paired robustness experiment.
        if any(t in q for t in ("ảnh nhỏ","thu nhỏ","resize_96","kích thước nhỏ")):
            data = self.structured_engine.get_robustness_summary()
            v = data["variants"]["resize_96_jpeg_q85"]
            return {"type":"robustness_small_image","explanation":
                    f"Thử nghiệm trên {data['source_samples']} ảnh nguồn: khi thu nhỏ 96x96 rồi nén JPEG Q85, "
                    f"Top-1 giữ nguyên {v['top1_preserved']}; trạng thái giữ nguyên {v['status_preserved']}; "
                    f"accuracy {v['accuracy']}; độ đổi OOD trung bình {v['delta_ood_mean']:.4f}.",
                    "source_file":data["source_ref"]}
        if "shiba" in q:
            data = self.structured_engine.get_shiba_inu_evidence()
            breed = data["breed_results"]; group = data["group_results"]
            return {"type":"shiba_inu","details":data,"explanation":
                    "Shiba Inu không nằm trong 120 giống hỗ trợ. "
                    f"Riêng {data['breed_total_images']} ảnh Shiba Inu: DOG {breed.get('DOG_rate')}, "
                    f"UNCERTAIN {breed.get('UNCERTAIN_rate')}. "
                    f"Toàn nhóm {data['group_total_images']} ảnh của 4 giống ngoài danh sách: DOG {group.get('DOG_rate')}, "
                    f"UNCERTAIN {group.get('UNCERTAIN_rate')}. Hai mẫu số này khác nhau.",
                    "source_file":data["source_ref"]}
        if "phú quốc" in q or "phu quoc" in q:
            data = self.structured_engine.get_phu_quoc_evidence()
            return {"type":"phu_quoc","details":data,"explanation":
                    f"Kho dữ liệu có {data['wikimedia_images_count']} ảnh khảo sát chó Phú Quốc. "
                    "Ánh xạ lớp đang triển khai không có nhãn phu_quoc_ridgeback; "
                    "không có bằng chứng kết quả phân loại đúng giống Phú Quốc từ model hiện tại. "
                    "Sự có mặt của ảnh trong dự án không chứng minh model đã được huấn luyện với chúng.",
                    "source_file":data["source_ref"]}

        # 9. Top confusions for a specific breed (supports explicit mention or conversation pronoun)
        target_breed = None
        for b in ["collie", "siberian_husky", "husky", "miniature_poodle", "poodle", "standard_schnauzer", "schnauzer", "kelpie", "rhodesian_ridgeback", "cardigan"]:
            if b in q:
                target_breed = "siberian_husky" if b == "husky" else ("collie" if b == "collie" else b)
                break

        # Check conversation history for breed pronoun
        if not target_breed and history and any(p in q for p in ["nó", "giống đó", "giống này", "con này"]):
            for turn in reversed(history[-4:]):
                t_content = turn.get("content", "").lower()
                for b in ["siberian_husky", "collie", "malamute", "eskimo_dog", "poodle", "schnauzer", "husky"]:
                    if b in t_content:
                        target_breed = "siberian_husky" if b == "husky" else b
                        break
                if target_breed:
                    break

        if target_breed and ("nhầm" in q or "sai" in q or "confusion" in q or "cặp" in q or "dự đoán thành" in q):
            confusions = self.structured_engine.get_top_confusions(breed_name=target_breed, top_n=5)
            if confusions:
                pairs_str = "; ".join([
                    f"{c['true_breed']} bị nhầm thành {c['predicted_breed']} ({c['count']} ảnh, chiếm {c['share_of_true_class']*100:.1f}% lớp thật)"
                    for c in confusions if target_breed in c['true_breed']
                ]) or "; ".join([f"{c['true_breed']} -> {c['predicted_breed']} ({c['count']} ảnh)" for c in confusions])
                return {
                    "type": "breed_confusions",
                    "breed": target_breed,
                    "confusions": confusions,
                    "explanation": (
                        f"Trong kết quả thực nghiệm đã lưu (outputs/quality_evaluation_2026-10-16/top_confusions.csv), "
                        f"các trường hợp nhầm lẫn đối với '{target_breed}' gồm: {pairs_str}."
                    ),
                    "source_file": "outputs/quality_evaluation_2026-10-16/top_confusions.csv",
                }

        return None

    def retrieve(self, query: str, history: Optional[List[Dict[str, str]]] = None, top_k: Optional[int] = None) -> RetrievalResult:
        """Executes hybrid retrieval: routes structured data and fetches relevant document chunks."""
        k = top_k or self.config.retrieval.top_k
        expanded_query = self.expand_query(query, history=history)

        # 1. Check structured analytical route
        structured_info = self.route_structured_query(query, history=history)
        if structured_info and hasattr(self.indexer,'source_chunks'):
            path = structured_info.get('source_file','').split('#')[0]
            if hasattr(self.indexer, 'db_path'):
                terms = expanded_query + ' ' + json.dumps(structured_info, ensure_ascii=False)
                chunks = [c for c, _ in self.indexer.search(terms, top_k=min(k, 2), source_paths=[path])]
            else:
                chunks = []
            chunks = chunks or self.indexer.source_chunks(path,min(k,2))
            sources = [{'chunk_id':c.chunk_id,'source_path':c.source_path,'title':c.title,
                        'start_line':c.start_line,'end_line':c.end_line,'relevance_score':1.0,
                        'locator':c.metadata.get('locator',''),'page':c.metadata.get('page'),
                        'row':c.metadata.get('row'),'json_pointer':c.metadata.get('json_pointer')} for c in chunks]
            return RetrievalResult(query,expanded_query,chunks,structured_info,sources)

        # 2. Search inverted index
        translated_query = self.query_rewriter.rewrite(query)
        q = query.lower()
        paper_question = any(t in q for t in ('paper','bài báo'))
        comparison = paper_question and any(t in q for t in ('so sánh','khác','đối chiếu','compare')) and any(t in q for t in ('dự án','hiện tại','đang triển khai','project'))
        architecture_question = any(t in q for t in ('kiến trúc', 'architecture', 'embedding', 'backbone'))
        deployment_question = ('phiên bản' in q or 'model_version' in q or 'checkpoint' in q) and any(
            t in q for t in ('mô hình', 'model', 'triển khai', 'deploy'))
        api_question = any(t in q for t in ('api', 'upload', 'tải lên'))
        current_question = not paper_question and (architecture_question or deployment_question or api_question)
        raw_results = [] if paper_question or current_question else self._search(expanded_query, translated_query, top_k=k * 8)
        # A test dataset is not a software test. Keep RAG prompts and fixtures out
        # of general evidence unless the user actually asks about that software.
        software_question = any(t in q for t in ('rag', 'retriev', 'truy xuất', 'unittest',
            'pytest', 'unit test', 'regression', 'hồi quy', 'kiểm thử mã', 'kiểm thử phần mềm'))
        if not software_question:
            raw_results = [(c,s) for c,s in raw_results if not c.source_path.startswith(('src/rag/','tests/','static/','templates/'))
                           and c.source_path not in ('docs/RAG.md', 'run_rag.py', 'run_rag.bat', 'configs/rag_config.yaml')]
        # Current implementation questions need evidence from the deployed sources.
        project_paths = ['README.md','outputs/deployment_oe_v2/model_manifest.json','src/inference/oe_v2_predictor.py']
        if api_question:
            project_paths = ['app.py', 'README.md', 'src/inference/preprocessing.py']
        if any(t in q for t in ('accuracy','chính xác','kết quả','top-1','top-5')):
            project_paths = ['outputs/quality_evaluation_2026-10-16/REPORT_VI.md','outputs/quality_evaluation_2026-10-16/summary.json']
        scoped = []
        if hasattr(self.indexer,'db_path') and (comparison or current_question):
            project_query = 'accuracy top1 top5 top1_accuracy top5_accuracy supported_id test_images_count' if comparison and any(t in q for t in ('accuracy','chính xác','top-1','top-5')) else expanded_query
            if api_question and not paper_question:
                project_query += ' multipart request.files image MAX_CONTENT_LENGTH MAX_UPLOAD_FILE_BYTES'
            scoped = self._search(project_query, translated_query, top_k=k*2, source_paths=project_paths)
            if current_question:
                scoped = [(c,s) for c,s in scoped if 'historical baseline' not in c.title.lower()]
            scoped = scoped[:k]
            raw_results = scoped + raw_results
            if not raw_results and not paper_question:
                raw_results = self._search(expanded_query, translated_query, top_k=k*8)
        if paper_question:
            paper_results = (self._search(expanded_query, translated_query, top_k=k*4, paper_only=True)
                             if hasattr(self.indexer,'db_path') else [(c,s) for c,s in raw_results if c.doc_type=='paper'])
            if comparison:
                project_results = scoped or [(c,s) for c,s in raw_results if c.doc_type not in ('paper','paper_note')]
                # Reserve evidence for both sides instead of filling every slot with the paper.
                raw_results = [item for pair in zip(paper_results,project_results) for item in pair] + paper_results + project_results
            elif paper_results:
                raw_results = paper_results
        elif current_question or any(t in q for t in ('dự án', 'api', 'triển khai', 'hiện tại')):
            raw_results = [(c,s) for c,s in raw_results if c.doc_type not in ('paper','paper_note')]

        # Questions about split independence need observed audit results, not
        # assertions in training code or reports for older candidate datasets.
        dataset_overlap = (not paper_question and not software_question
            and any(t in q for t in ('trùng', 'duplicate', 'overlap', 'leakage', 'rò rỉ'))
            and any(t in q for t in ('dữ liệu', 'dataset', 'train', 'test', 'ảnh', 'huấn luyện')))
        if dataset_overlap and hasattr(self.indexer, 'db_path'):
            audit_dir = self.structured_engine.qual_dir.relative_to(self.config.project_root).as_posix()
            audit_query = 'trùng duplicate near_duplicate leakage overlap same_source_family'
            audit = self.indexer.search(audit_query, top_k=1,
                source_paths=[audit_dir + '/REPORT_VI.md'])
            if audit:
                details = self.indexer.search(audit_query + ' likely_same_source_family', top_k=k,
                    source_paths=[audit_dir + '/near_duplicate_visual_review.csv'])
                raw_results = audit + details + raw_results

        # 3. Post-filter and rank chunks
        selected_chunks: List[DocumentChunk] = []
        sources: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        # If structured query matched a specific source file, boost those chunks first
        priority_file = structured_info.get("source_file", "").split("#")[0] if structured_info else None

        for chunk, score in raw_results:
            if chunk.chunk_id in seen_ids:
                continue

            # Check priority boost
            is_priority = priority_file and priority_file in chunk.source_path
            # FTS5 BM25 and the legacy in-memory scorer have different scales.
            minimum = 0.0 if hasattr(self.indexer, 'db_path') else self.config.retrieval.min_relevance_score
            if score > minimum or is_priority:
                selected_chunks.append(chunk)
                seen_ids.add(chunk.chunk_id)
                sources.append({
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "title": chunk.title,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "relevance_score": round(score, 3),
                    "locator": chunk.metadata.get('locator', ''),
                    "page": chunk.metadata.get('page'),
                    "row": chunk.metadata.get('row'),
                    "json_pointer": chunk.metadata.get('json_pointer'),
                })

            if len(selected_chunks) >= k:
                break

        # If priority file wasn't matched in raw search, search by its filename directly
        if priority_file and not any(priority_file in c.source_path for c in selected_chunks):
            file_chunks = (self.indexer.source_chunks(priority_file,2) if hasattr(self.indexer,'source_chunks')
                           else [c for c in self.indexer.chunks if priority_file in c.source_path])
            for fc in file_chunks[:2]:
                if fc.chunk_id not in seen_ids:
                    selected_chunks.insert(0, fc)
                    seen_ids.add(fc.chunk_id)
                    sources.insert(0, {
                        "chunk_id": fc.chunk_id,
                        "source_path": fc.source_path,
                        "title": fc.title,
                        "start_line": fc.start_line,
                        "end_line": fc.end_line,
                        "relevance_score": 1.0,
                        "locator": fc.metadata.get('locator', ''),
                        "page": fc.metadata.get('page'),
                    })

        return RetrievalResult(
            query=query,
            expanded_query=expanded_query,
            chunks=selected_chunks[:k],
            structured_data=structured_info,
            sources=sources[:k],
        )
