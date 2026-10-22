# Paper trong kho kiến thức

Đặt các paper PDF, bản thảo DOCX, TXT, Markdown, LaTeX hoặc BibTeX vào thư mục này
(hoặc một thư mục khác bên trong dự án), sau đó chạy:

```powershell
.\.venv\Scripts\python.exe run_rag.py --update
```

PDF có lớp văn bản được đọc theo trang. PDF scan hoặc trang không trích được chữ
được ghi `ocr_required` trong `outputs/rag_index/coverage.json`; chúng chưa được
coi là đã đọc đầy đủ. Hãy OCR thành PDF có lớp văn bản rồi cập nhật lại.

Chỉ có liên kết hoặc mục BibTeX chưa đồng nghĩa đã đọc toàn văn paper. Bộ nạp không
tự tải tài liệu từ Internet. Hình, sơ đồ và công thức dạng ảnh chưa được phân tích
thị giác; chatbot phải nêu giới hạn khi văn bản không đủ bằng chứng.

Paper đã thêm: `dog-breed-recognition.pdf`, *An Intelligent Dog Breed Recognition
System Using Deep Learning* (2022), 14 trang. Tệp `.metadata.json` đi kèm lưu tên
và tác giả đã đối chiếu; `.notes.md` bổ sung ghi chú đọc hình tại trang 9–12.
