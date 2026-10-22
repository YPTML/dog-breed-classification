"""Real ingestion/retrieval regressions on a small isolated project corpus."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.rag.config import RAGConfig
from src.rag.indexer import Indexer, tokenize
from src.rag.knowledge_base import KnowledgeBase
from src.rag.project_documents import ProjectReader, discover, permitted
from src.rag.llm_client import LLMClient, LLMResponse
from src.rag.retriever import HybridRetriever
from src.rag.query_rewriter import LocalQueryRewriter


class ProjectCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = RAGConfig(project_root=self.root)
        self.config.knowledge_base.discover_project = True
        self.config.knowledge_base.index_dir = self.root / 'index'
        self.config.llm.provider = 'offline_grounded'

    def write(self,path,text):
        target=self.root/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(text,encoding='utf-8')
        return target

    def test_discovery_includes_new_nested_documents_and_excludes_secrets(self):
        self.write('papers/new-note.md','Novel architecture explanation')
        self.write('data/labels.csv','breed,count\nhusky,8\n')
        self.write('.env.local','API_KEY=never-index-this')
        self.write('secrets.json','{"key":"secret"}')
        self.write('.venv/irrelevant.py','library text')
        self.write('index/cache.json','do not reindex index')
        found,_=discover(self.config)
        self.assertEqual(found,['data/labels.csv','papers/new-note.md'])
        self.assertFalse(permitted(self.config,'../outside.md'))

    def test_csv_tail_and_long_json_are_searchable_with_exact_locators(self):
        self.write('data.csv','id,note\n'+'\n'.join(f'{i},ordinary' for i in range(40))+'\n40,zebramarker\n')
        self.write('big.json',json.dumps({'rows':[{'detail':'x'*3000},{'detail':'quartzmarker'}]}))
        index=Indexer(self.config);index.rebuild()
        csv_result=index.search('zebramarker',1)[0][0]
        json_result=index.search('quartzmarker',1)[0][0]
        self.assertEqual(csv_result.metadata['row'],41)
        self.assertIn('/rows/1/detail',json_result.content)
        self.assertTrue(all(len(c.content)<=self.config.chunking.markdown_max_tokens*5 for c in index.chunks))

    def test_updates_remove_old_chunks_and_deleted_files(self):
        source=self.write('notes.md','olduniquetoken')
        index=Indexer(self.config);index.rebuild()
        version=index.knowledge_version
        source.write_text('newuniquetoken',encoding='utf-8')
        index.update_index()
        self.assertNotEqual(version,index.knowledge_version)
        self.assertEqual(index.search('olduniquetoken'),[])
        self.assertTrue(index.search('newuniquetoken'))
        source.unlink();index.update_index()
        self.assertEqual(len(index.chunks),0)

    def test_reader_can_start_and_search_while_update_is_uncommitted(self):
        self.write('notes.md','committedmarker')
        index=Indexer(self.config);index.rebuild()
        with index.connect() as writer:
            writer.execute("INSERT OR REPLACE INTO settings VALUES ('update_in_progress','yes')")
            reader=Indexer(self.config)
            self.assertTrue(reader.search('committedmarker'))

    def test_parser_error_is_reported_without_partial_stale_rows(self):
        self.write('broken.json','{not json')
        index=Indexer(self.config);result=index.rebuild()
        self.assertEqual(result['issues_count'],1)
        self.assertEqual(index.coverage['files'][0]['status'],'error')
        self.assertEqual(len(index.chunks),0)

    def test_real_pdf_text_and_blank_scanned_page_report(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject
        path=self.root/'paper.pdf'
        writer=PdfWriter();page=writer.add_blank_page(width=600,height=800)
        font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        stream=DecodedStreamObject();stream.set_data(b'BT /F1 12 Tf 40 700 Td (Zebrapaper uses inverted residual bottleneck layers.) Tj ET')
        page[NameObject('/Contents')]=writer._add_object(stream)
        writer.add_blank_page(width=600,height=800)
        with path.open('wb') as f:writer.write(f)
        index=Indexer(self.config);index.rebuild()
        chunk=index.search('Zebrapaper',1)[0][0]
        self.assertEqual(chunk.metadata['page'],1)
        self.assertEqual(chunk.doc_type,'paper')
        self.assertEqual(index.coverage['files'][0]['issues'][0]['status'],'ocr_required')
        self.assertEqual(index.coverage['files'][0]['status'],'partial')
        retrieved=HybridRetriever(index,self.config).retrieve('Paper Zebrapaper uses what layers?')
        self.assertTrue(any(c.doc_type=='paper' for c in retrieved.chunks))
        self.assertEqual(retrieved.sources[0]['page'],1)

    def test_missing_prediction_score_is_not_reported_as_zero(self):
        answer=LLMClient(self.config.llm).generate('Giải thích kết quả ảnh này', [],
            prediction_context={'status':'DOG','breed':'husky','confidence':0.9})
        self.assertIn('điểm OOD chưa cung cấp',answer.answer)
        self.assertNotIn('0.000000',answer.answer)

    def test_llm_citations_resolve_to_exact_pdf_page(self):
        client=LLMClient(self.config.llm)
        chunks=[{'source_path':'paper.pdf','metadata':{'locator':'page=11&L1-L8'}}]
        resolved=client._resolve_citations(LLMResponse('Accuracy 81% [S1].','live','ollama','test'),chunks)
        self.assertIn('[paper.pdf#page=11&L1-L8]',resolved.answer)
        invalid=client._resolve_citations(LLMResponse('Accuracy 99% [S9].','live','ollama','test'),chunks)
        self.assertEqual(invalid.llm_status,'error')
        self.assertNotIn('99%',invalid.answer)

    def test_grouped_llm_citations_are_resolved_and_validated(self):
        client = LLMClient(self.config.llm)
        chunks = [
            {'source_path': 'paper.pdf', 'metadata': {'locator': 'page=9'}},
            {'source_path': 'model.py', 'start_line': 12, 'end_line': 20},
        ]
        result = client._resolve_citations(
            LLMResponse('Embedding 1280 [S1, S2]; classes 120 [S2].', 'live', 'ollama', 'test'), chunks)
        self.assertEqual(result.llm_status, 'live')
        self.assertNotIn('[S', result.answer)
        self.assertIn('[paper.pdf#page=9]', result.answer)
        self.assertIn('[model.py#L12-L20]', result.answer)
        invalid = client._resolve_citations(
            LLMResponse('Unsupported fact [S1, S99].', 'live', 'ollama', 'test'), chunks)
        self.assertEqual(invalid.llm_status, 'error')

    def test_local_generation_renders_structured_citations(self):
        client = LLMClient(self.config.llm)
        with patch('src.rag.llm_client.requests.post') as post:
            post.return_value.json.return_value = {'message': {'content': json.dumps(
                {'paragraphs': [{'text': 'Nguồn không ghi đơn vị. [S1]', 'source_ids': [1]}]})}}
            result = client._call_ollama('Evidence and question', source_count=2)
        self.assertEqual(result.llm_status, 'live')
        self.assertEqual(result.answer.count('[S1]'), 1)
        schema = post.call_args.kwargs['json']['format']
        ids = schema['properties']['paragraphs']['items']['properties']['source_ids']
        self.assertEqual(ids['items']['enum'], [1, 2])

    def test_local_generation_rejects_invalid_structured_citations(self):
        client = LLMClient(self.config.llm)
        for source_ids in ([], [99], [True], ['1']):
            with self.subTest(source_ids=source_ids), patch('src.rag.llm_client.requests.post') as post:
                post.return_value.json.return_value = {'message': {'content': json.dumps(
                    {'paragraphs': [{'text': 'Unverified answer', 'source_ids': source_ids}]})}}
                result = client._call_ollama('Evidence and question', source_count=1)
            self.assertEqual(result.llm_status, 'error')
            self.assertNotIn('Unverified answer', result.answer)

    def test_paper_metrics_are_not_an_inventory_question(self):
        self.write('papers/study.notes.md','Paper accuracy validation results: 81 percent.')
        self.write('README.md','Current architecture MobileNetV2 embedding 1280.')
        self.write('outputs/quality_evaluation_2026-10-16/REPORT_VI.md','Project accuracy top1: 83.58 percent.')
        index=Indexer(self.config);index.rebuild()
        # This fixture represents a corpus with an already indexed paper.
        index.coverage['paper_files']=1
        retriever=HybridRetriever(index,self.config)
        r=retriever.retrieve('Paper đạt độ chính xác bao nhiêu?')
        self.assertIsNone(r.structured_data)
        self.assertTrue(all(c.doc_type in ('paper','paper_note') for c in r.chunks))
        r=retriever.retrieve('So sánh độ chính xác trong paper và mô hình dự án hiện tại')
        self.assertTrue(any(c.source_path.startswith('papers/') for c in r.chunks))
        self.assertTrue(any(c.source_path.endswith('REPORT_VI.md') for c in r.chunks))
        r=retriever.retrieve('Kiến trúc mô hình hiện tại?')
        self.assertEqual(r.chunks[0].source_path,'README.md')
        self.assertFalse(any(c.source_path.startswith('papers/') for c in r.chunks))

    def test_current_dataset_question_is_not_limited_to_model_sources(self):
        self.write('README.md', 'Training train epochs optimizer test unittest.')
        self.write('reports/duplicates.md', 'Dữ liệu có ảnh trùng giữa tập huấn luyện và kiểm thử.')
        index = Indexer(self.config); index.rebuild()
        result = HybridRetriever(index, self.config).retrieve(
            'Dữ liệu hiện tại có bị trùng giữa tập huấn luyện và kiểm thử không?')
        self.assertTrue(any(c.source_path == 'reports/duplicates.md' for c in result.chunks))

    def test_dataset_overlap_prefers_evaluation_audit_over_test_code(self):
        self.write('README.md', 'Dữ liệu hiện tại huấn luyện kiểm thử training testing train test unittest.')
        self.write('tests/test_split.py', 'Dữ liệu hiện tại có bị trùng giữa tập huấn luyện và kiểm thử không?')
        report = 'outputs/quality_evaluation_2026-10-16/REPORT_VI.md'
        self.write(report, '# Audit\n\nExact-overlap checks: 0. Near-duplicate review found likely_same_source_family pairs.')
        self.write('outputs/reports/OLD_AUDIT.md', 'Exact duplicate training testing dataset overlap checks were clean.')
        index = Indexer(self.config); index.rebuild()
        retriever = HybridRetriever(index, self.config)
        with patch.object(retriever.query_rewriter, 'rewrite', return_value='overlaps between training and testing datasets currently'):
            result = retriever.retrieve('Dữ liệu hiện tại có bị trùng giữa tập huấn luyện và kiểm thử không?', top_k=2)
        self.assertEqual(result.chunks[0].source_path, report)
        self.assertFalse(any(c.source_path.startswith('tests/') for c in result.chunks))

    def test_translated_query_finds_english_paper_passage(self):
        self.write('papers/study.notes.md', 'Blurred images or images taken during night time were not represented in our results.')
        self.write('papers/other.notes.md', 'Ảnh trong bài báo có các hình kiến trúc và cấu hình huấn luyện.')
        index = Indexer(self.config); index.rebuild(); index.coverage['paper_files'] = 1
        retriever = HybridRetriever(index, self.config)
        with patch.object(retriever.query_rewriter, 'rewrite', return_value='blurred images night time evaluation'):
            result = retriever.retrieve('Paper có đánh giá ảnh mờ hoặc ảnh chụp ban đêm không?')
        self.assertTrue(any('night time' in c.content for c in result.chunks))

    def test_translation_keeps_vietnamese_notes_in_small_context(self):
        self.write('papers/study.notes.md', 'Bộ nhớ sau tối ưu ghi 965 nhưng không ghi đơn vị.')
        for number in range(5):
            self.write(f'papers/english{number}.notes.md', 'Memory optimization table. Original memory consumption 1.2 GB.')
        index = Indexer(self.config); index.rebuild(); index.coverage['paper_files'] = 1
        retriever = HybridRetriever(index, self.config)
        with patch.object(retriever.query_rewriter, 'rewrite', return_value='memory optimization consumption table unit'):
            result = retriever.retrieve('Paper báo cáo bộ nhớ sau tối ưu và đơn vị là gì?', top_k=2)
        self.assertTrue(any('không ghi đơn vị' in c.content for c in result.chunks))
        self.assertTrue(any('Memory optimization' in c.content for c in result.chunks))

    def test_query_rewriter_uses_only_loopback_and_caches_valid_results(self):
        self.config.llm.provider = 'ollama'
        self.config.llm.model = 'local-test'
        self.config.llm.api_base = 'http://127.0.0.1:11434'
        rewriter = LocalQueryRewriter(self.config)
        with patch('src.rag.query_rewriter.requests.post') as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {'message': {'content': '{"search_query":"blurred images night time"}'}}
            self.assertEqual(rewriter.rewrite('Ảnh mờ ban đêm?'), 'blurred images night time')
            self.assertEqual(rewriter.rewrite('Ảnh mờ ban đêm?'), 'blurred images night time')
            self.assertEqual(post.call_count, 1)
            self.assertFalse(post.call_args.kwargs['allow_redirects'])
            self.assertNotIn('documents', post.call_args.kwargs['json'])
            self.config.llm.api_base = 'https://127.0.0.1.example.com'
            self.assertEqual(rewriter.rewrite('Câu hỏi khác'), '')
            self.assertEqual(post.call_count, 1)

    def test_query_rewriter_failure_preserves_lexical_fallback(self):
        self.config.llm.provider = 'ollama'
        rewriter = LocalQueryRewriter(self.config)
        with patch('src.rag.query_rewriter.requests.post') as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {'message': {'content': 'not JSON'}}
            self.assertEqual(rewriter.rewrite('Câu hỏi'), '')
            post.return_value.json.return_value = {'message': {'content': '{"search_query":[]}'}}
            self.assertEqual(rewriter.rewrite('Câu hỏi'), '')
            self.config.llm.provider = 'offline_grounded'
            self.assertEqual(rewriter.rewrite('Câu hỏi'), '')
            self.assertEqual(post.call_count, 2)

    def test_purpose_answer_uses_overview_instead_of_bugfix_table(self):
        self.write('README.md','An image classifier predicts the breed from an uploaded image.')
        self.write('BUGFIX_REPORT.md','| API | errors | fixed |')
        self.write('outputs/deployment_oe_v2/class_names.json',json.dumps({'0':'husky'}))
        index=Indexer(self.config);index.rebuild()
        retrieved=HybridRetriever(index,self.config).retrieve('Mô hình dùng cho mục đích gì?')
        answer=LLMClient(self.config.llm).generate('Mô hình dùng cho mục đích gì?',
            [c.to_dict() for c in retrieved.chunks],retrieved.structured_data)
        self.assertIn('nhận diện giống chó từ ảnh',answer.answer)
        self.assertIn('README.md',answer.answer)
        self.assertNotIn('BUGFIX_REPORT',answer.answer)

    def test_notebook_and_docx_are_not_executed(self):
        from docx import Document
        self.write('experiment.ipynb',json.dumps({'cells':[{'cell_type':'markdown','source':['Notebookmarker result']},
            {'cell_type':'code','source':['raise RuntimeError("must not execute")'],'outputs':[]}]}))
        doc=Document();doc.add_paragraph('Docxmarker conclusion');doc.save(self.root/'paper.docx')
        index=Indexer(self.config);index.rebuild()
        self.assertTrue(index.search('Notebookmarker'))
        self.assertTrue(index.search('Docxmarker'))

    def test_model_training_is_not_misclassified_as_pet_care(self):
        chunk={'content':'Training uses learning rate schedules.', 'source_path':'training.md','title':'Training',
               'start_line':1,'end_line':1}
        answer=LLMClient(self.config.llm).generate('Quy trình huấn luyện mô hình?', [chunk])
        self.assertIn('Training uses',answer.answer)
        self.assertNotIn('chăm sóc',answer.answer)
        self.assertEqual(answer.llm_status,'missing_config')

    def test_unicode_tokenizer_preserves_vietnamese_words(self):
        self.assertIn('huấn',tokenize('huấn luyện mô hình'))
        self.assertIn('luyện',tokenize('huấn luyện mô hình'))

    def test_secret_values_not_returned_in_source_excerpt(self):
        self.write('config.txt','api_key = super-secret-value-0123456789\nhttps://example.com?key=topsecret0123456789')
        index=Indexer(self.config);index.rebuild()
        text='\n'.join(c.content for c in index.chunks)
        self.assertNotIn('super-secret-value',text)
        self.assertNotIn('topsecret0123456789',text)
        self.assertIn('[REDACTED]',text)


if __name__=='__main__':unittest.main()
