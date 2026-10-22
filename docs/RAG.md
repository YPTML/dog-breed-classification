# Hỏi đáp toàn bộ kho tài liệu dự án

## Cài đặt và cập nhật

Trên máy Windows này, chạy `run_rag.bat` để mở ứng dụng. Launcher ưu tiên môi trường
riêng tại `%LOCALAPPDATA%\DogBreedRAG\venv`, đã chuẩn bị để tránh việc đọc thư viện
chậm trên ổ E; nếu không có thì dùng `.venv` của dự án. Có thể đặt `RAG_PYTHON`
thành đường dẫn `python.exe` mong muốn. Dùng `run_rag.bat --update` để cập nhật
tri thức, hoặc `run_rag.bat --cli` để hỏi trong terminal mà không nạp TensorFlow.
Khi đã có chỉ mục, khởi động thông thường sử dụng ngay chỉ mục đó, không quét lại
toàn bộ dữ liệu ảnh. Chạy `--update` sau khi thêm, sửa hoặc xóa tài liệu.

Trên Windows, SQLite được lưu tại `%LOCALAPPDATA%\DogBreedRAG\indexes\<mã-dự-án>`
để giảm đọc ngẫu nhiên trên ổ chứa dữ liệu. Mỗi đường dẫn dự án có chỉ mục riêng.
`--status`/API trạng thái cho biết vị trí thực; báo cáo phạm vi vẫn được xuất vào
`outputs/rag_index/coverage.json`. Có thể đặt `RAG_INDEX_DIR` để chọn vị trí khác,
hoặc tắt `knowledge_base.use_local_appdata` để dùng `index_dir` trong YAML.

Các lệnh tương đương khi dùng môi trường `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rag.txt
.\.venv\Scripts\python.exe run_rag.py --update
.\.venv\Scripts\python.exe run_rag.py --status
.\.venv\Scripts\python.exe run_rag.py
```

Không thay đổi trọng số, centroid hay các ngưỡng của bộ phân loại ảnh. Các thay đổi
tài liệu cần được cập nhật bằng `--update`; tệp thêm/sửa/xóa được đồng bộ vào chỉ mục.
Máy chủ đang chạy sẽ đọc phiên bản chỉ mục mới ở lần hỏi tiếp theo. Khi thay mã nguồn
Python, cần khởi động lại máy chủ để dùng mã mới.
`run_rag.py` giữ cổng ngay trước khi nạp chỉ mục và mô hình. Nếu cổng đang bận,
lần chạy thứ hai thoát với mã 1 và chỉ dẫn dùng `--port`; máy chủ đang chạy
không bị dừng. Trên Windows, socket dùng chế độ độc quyền để tránh hai phiên
cùng nhận yêu cầu trên một cổng. Cổng được giải phóng khi khởi động thất bại.

## Phạm vi nguồn

`knowledge_base.discover_project: true` quét cả nguồn bị Git ignore trong dự án,
không phụ thuộc danh sách tệp cũ. Hỗ trợ Markdown, text, mã nguồn, HTML/CSS,
cấu hình, XML, BibTeX/LaTeX, CSV/TSV, JSON/JSONL, notebook, PDF và DOCX.

- CSV được đọc toàn bộ các dòng vào SQLite; JSON được chia theo JSON pointer,
  không cắt còn 15 dòng hoặc 3.500 ký tự. Notebook chỉ đọc, không chạy cell.
- PDF giữ số trang, tiêu đề và tác giả có trong metadata; DOCX giữ văn bản và bảng.
- Model Keras đọc cấu hình/metadata trong archive, không chạy deserialize model.
  NPY chỉ đọc shape/dtype; NPZ chỉ liệt kê thành phần, không nhúng toàn bộ tensor.
- Ảnh được thống kê theo thư mục; nhãn và các kết quả có trong manifest/CSV được
  lập chỉ mục. Không khẳng định chatbot đã hiểu pixel của tất cả ảnh.
- Khóa, thông tin xác thực, virtualenv, thư viện, cache và chính chỉ mục RAG bị loại.
  Giá trị bí mật nhận diện được trong tài liệu cũng được che trước khi lưu.
- Tệp lớn hơn giới hạn cấu hình, định dạng chưa hỗ trợ, lỗi trích xuất và PDF cần
  OCR được công khai trong báo cáo phạm vi, không báo là đã đọc hoàn chỉnh.

SQLite FTS5 sử dụng tìm kiếm từ khóa Unicode/BM25, mở rộng thuật ngữ Việt–Anh và
định tuyến dữ liệu có cấu trúc. Đây chưa phải bộ embedding ngữ nghĩa đa ngôn ngữ.
Các số liệu báo cáo được đọc từ tệp nguồn thay vì lưu cứng trong câu trả lời.
Với Ollama cục bộ, câu hỏi tự do còn được chuyển thành từ khóa tiếng Anh để tìm
paper/mã nguồn, rồi hợp nhất thứ hạng với truy vấn gốc. Bước này chỉ gửi câu hỏi
tới loopback, không gửi tài liệu và không dùng kết quả dịch làm bằng chứng.
Kết quả tốt nhất của cả truy vấn gốc và bản dịch được giữ lại để tránh bỏ mất
ghi chú tiếng Việt khi tìm paper tiếng Anh.
Nếu Ollama không hoạt động, hệ thống vẫn tìm theo từ khóa gốc. Có thể tắt bằng
`retrieval.local_query_rewrite: false`; các câu hỏi số liệu xác định không cần dịch.
Với câu hỏi về trùng dữ liệu train/test, ưu tiên báo cáo chất lượng đang dùng
và kết quả rà soát gần trùng. Kiểm tra trùng byte và kiểm tra cùng nguồn ảnh là
hai loại bằng chứng khác nhau; không suy ra dữ liệu sạch chỉ từ mã chia tập.

## Paper

Thêm file vào `papers/`, chạy cập nhật rồi hỏi về nội dung. Paper là nguồn nghiên cứu;
phương pháp có trong paper không tự chứng minh code dự án đã triển khai phương pháp đó.
Trang scan thiếu lớp chữ cần OCR trước. Mở nguồn ở mỗi câu trả lời để xem trích đoạn
đã dùng, cùng trang PDF, dòng dữ liệu CSV hoặc JSON pointer.

## Chọn LLM

Mặc định dự án dùng **Ollama tại máy**, model `qwen3:4b-instruct`, địa chỉ
`http://127.0.0.1:11434`. Tải model một lần bằng `ollama pull qwen3:4b-instruct`.
`run_rag.py` thử khởi động Ollama đã cài nếu dịch vụ chưa chạy. Câu hỏi và trích đoạn
được gửi tới địa chỉ loopback này; không sử dụng API đám mây với cấu hình mặc định.
Lần hỏi đầu tiên có thể lâu hơn vì phải nạp model vào bộ nhớ.
Khi có tài liệu tham khảo, Ollama trả các đoạn văn kèm mã nguồn theo JSON schema;
hệ thống kiểm tra mã trước khi hiển thị và chuyển thành liên kết tới trích đoạn.
Kiểm tra này xác nhận nguồn tồn tại, không chứng minh mọi diễn giải đều chính xác.

Khi chưa có LLM thật, giao diện ghi **Chế độ trích đoạn · chưa có LLM**. Hệ thống vẫn
tra cứu văn bản và trả lời truy vấn số liệu xác định; không giả danh là tổng hợp LLM.

Với API: đặt `GEMINI_API_KEY`, `OPENROUTER_API_KEY` hoặc `OPENAI_API_KEY` thật trong
môi trường tiến trình, chọn provider/model tương ứng trong cấu hình hoặc biến
`RAG_LLM_PROVIDER` và `RAG_LLM_MODEL`. Khóa mẫu không được coi là đã cấu hình.
Khi dùng API, chỉ các đoạn truy xuất, câu hỏi và lịch sử gần được gửi tới provider.

Với Ollama đã cài và có model:

```powershell
$env:RAG_LLM_PROVIDER = 'ollama'
$env:RAG_LLM_MODEL = '<tên-model-đã-cài>'
$env:OLLAMA_HOST = 'http://127.0.0.1:11434'
.\.venv\Scripts\python.exe run_rag.py
```

API `/api/rag/status` phân biệt kho chỉ mục và cấu hình LLM. Một API key có mặt
chỉ xác nhận cấu hình; kết nối/model cần được kiểm chứng bằng truy vấn thật.
Không đưa khóa bí mật vào tài liệu dự án hoặc chat.

## Kiểm chứng

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_rag_project_corpus tests.test_rag_pipeline tests.test_rag_api tests.test_rag_launcher -v
```

`outputs/rag_index/coverage.json` lưu từng tệp, số chunk, trạng thái, lỗi/trang cần
OCR, ảnh theo thư mục và danh sách định dạng chưa hỗ trợ. Phiên bản kiến thức có hash
nguồn; số tệp được lập chỉ mục không đồng nghĩa tất cả định dạng đều đã hiểu hoàn toàn.
