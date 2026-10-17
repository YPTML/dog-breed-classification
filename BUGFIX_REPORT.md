# Bugfix audit — 2026-09-30

Cập nhật kiểm chứng: 2026-10-01.

## Các lỗi đã sửa

| Khu vực | Lỗi và ảnh hưởng | Bản sửa |
| --- | --- | --- |
| API `/predict` | `ValueError`, `OSError`, `SyntaxError` từ mô hình bị trả thành HTTP 400 “Invalid or unsupported image”, không có stack trace. | Dùng `InvalidImageError` cho lỗi giải mã ảnh; lỗi mô hình trả HTTP 500 và ghi `logger.exception`. Nội dung lỗi nội bộ không gửi tới trình duyệt. |
| Giao diện | Chấp nhận kết quả DOG thiếu giống/xác suất; status kiểu số gây exception và thông báo nhầm lỗi mạng; top1 có thể mâu thuẫn với top5. | Kiểm tra kiểu status, trường dự đoán, thứ tự/độ trùng top5, tính nhất quán top1 và OOD score. Giữ tương thích kết quả cũ và NOT_DOG. |
| Tạo dữ liệu OOD | Dùng thư mục đích tùy chỉnh nhưng manifest quét thư mục mặc định; bỏ qua nguồn hard OOD mặc định khi có nguồn thường. | Dùng đúng đường dẫn đích trong bước lập và kiểm tra manifest; thu thập cả hai nguồn. |
| Tên ảnh OOD | Tên có hậu tố chống trùng vẫn có thể trùng tên ảnh khác và ghi đè trong cùng lượt tạo dữ liệu. | Tiếp tục thêm hậu tố cho tới khi tên chưa được sử dụng. |
| Bộ đọc ảnh OOD | Tìm thấy ảnh WebP nhưng dùng decoder TensorFlow không hỗ trợ định dạng này. NumPy `.item()` còn có thể bỏ byte NUL cuối ảnh: fixture 71.056 byte bị cắt còn 71.055 byte và không giải mã được. | Nhận dạng WebP bằng signature, giải mã RGB bằng Pillow; dùng `.tobytes()` cho NumPy fixed-width strings để giữ padding. Giữ nguyên bước resize và decoder của các định dạng cũ. Áp dụng cho 6 đường xử lý OOD. |
| OE V2 predictor | Đầu ra sai shape, xác suất sai miền/tổng, centroid không hữu hạn hoặc chỉ số lớp sai có thể đi tới bước dự đoán; NaN score bị phân loại thành NOT_DOG. | Kiểm tra dữ liệu lúc nạp và trước khi phân loại; từ chối score/ngưỡng không hợp lệ. Không thay ngưỡng đã hiệu chỉnh. |
| Test local API | Test giả lập `app.model` nhưng ứng dụng hiện gọi `predictor.inference_model`, khiến test không kiểm tra đúng đường thực thi. | Giả lập đúng mô hình hợp nhất, trả cả xác suất và embedding; cập nhật kiểm tra giới hạn ảnh và contract API hiện tại. |

## Kiểm chứng

- Test JavaScript: **48/48 đạt**.
- Test tạo dữ liệu/manifest: **15/15 đạt**.
- Unit test NumPy/Pillow: **8/8 đạt**, gồm kiểm tra shape, xác suất, centroid, chỉ số lớp, score/ngưỡng, ảnh lỗi/quá kích thước và bảo toàn byte WebP. Đã chạy các lớp test với TensorFlow import được giả lập và bỏ qua import các module không liên quan; không chạy graph TensorFlow hoặc mô hình thật trong lượt này.
- Tái hiện API trước/sau sửa: ba loại lỗi nội bộ chuyển từ HTTP 400, không log exception, sang HTTP 500 có log exception; ảnh không hợp lệ vẫn HTTP 400.
- Kiểm tra cú pháp Python và `git diff --check`: **đạt**.
- SHA-256 của mô hình OE V2 khớp giá trị đã khóa: `e4508564a9389250e265cae6ea6fed0e1a0edf8497d62cd9c8a44342bdd45e61`.
- **Chưa xác minh bộ test tích hợp với mô hình thật và `tf.data`.** Lượt kiểm tra ban đầu bị ngắt khi công cụ đóng; các lượt chạy lại mất nhiều phút nạp thư viện và lặp lại stack chờ nạp DLL TensorFlow. Đã dừng các tiến trình kiểm tra sau khi lưu log. Không coi log chưa hoàn tất là kết quả đạt. Cần chạy lại các lệnh bên dưới khi môi trường TensorFlow hoạt động bình thường.

Các kiểm tra dùng fixture tạm, ảnh tổng hợp và artifact hiện có; không huấn luyện lại mô hình.

## Log trong workspace

- `_audit_tmp/bugfix_api_isolated_before.log`: tái hiện lỗi phân loại HTTP trước sửa.
- `_audit_tmp/bugfix_api_isolated_after.log`: kết quả phân loại HTTP sau sửa.
- `_audit_tmp/bugfix_frontend_final.log`: 48 test JavaScript.
- `_audit_tmp/bugfix_data_final.log`: 15 test dữ liệu và manifest.
- `_audit_tmp/bugfix_validation_final.log`: 8 unit test NumPy/Pillow đạt; ghi rõ cách cô lập framework.
- `_audit_tmp/bugfix_python_final.log`: lượt thử toàn bộ `tests/test_*.py`, deployment smoke, Flask backend/frontend và local API; chưa hoàn tất.
- `_audit_tmp/bugfix_runtime_verified.log`: lượt kiểm tra runtime bị ngắt, **không phải kết quả đạt**.
- `_audit_tmp/bugfix_import_stacks.log`: chẩn đoán bước import TensorFlow chậm trong môi trường local.
- `_audit_tmp/bugfix_runtime_stacks.log`: stack trace lặp lại khi nạp thư viện native TensorFlow trong lượt chạy lại.

Log nằm trong thư mục đã được `.gitignore` loại trừ. Test hồi quy mới nằm trong `tests/test_api_bug_regressions.py`, `tests/test_data_bug_regressions.py`, `tests/test_inference_bug_regressions.py` và `src/test_frontend_behavior.js`.

Lệnh chạy lại:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe -m unittest src.test_deployment_smoke src.test_flask_backend src.test_flask_frontend src.test_local_app -v
node --test src/test_frontend_behavior.js
```
