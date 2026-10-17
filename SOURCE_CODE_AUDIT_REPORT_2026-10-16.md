# Báo cáo kiểm tra và sửa lỗi mã nguồn

**Dự án:** Dog Breed Classification  
**Ngày hoàn thành:** 2026-10-16  
**Phạm vi:** mã nguồn, cấu hình, điểm khởi chạy, API Flask, suy luận TensorFlow/OOD, giao diện JavaScript, công cụ dữ liệu và CLI OOD.

## Kết luận

Đã tái hiện và sửa hai lỗi còn tồn tại trong trạng thái mã nguồn hiện tại. Hai test hồi quy mới thất bại trên mã trước sửa và đạt sau sửa. Các bộ test chính, smoke test và test giao diện đều đạt sau thay đổi.

Báo cáo này không khẳng định chương trình hoàn toàn không còn lỗi. Việc huấn luyện lại mô hình, hiệu chỉnh lại ngưỡng và triển khai Google Cloud không được thực hiện trong lượt kiểm tra này.

## Lỗi đã xác minh

### 1. Ánh xạ chỉ số lớp có thể bị thay đổi mà vẫn vượt qua kiểm tra toàn vẹn

**Ảnh hưởng:** chương trình có thể khởi động bình thường nhưng trả sai tên giống chó so với chỉ số đầu ra của mô hình.

**Nguyên nhân gốc:** `verify_integrity=True` chỉ kiểm tra SHA-256 của model và centroid. `class_names.json` chỉ được kiểm tra đủ 120 chỉ số và tên không rỗng; việc hoán đổi hai nhãn hoặc dùng tên trùng không bị phát hiện.

**Bằng chứng tái hiện:** một ánh xạ đủ các chỉ số `0..119` nhưng có tên trùng tiếp tục tới bước nạp model. Một ánh xạ hoán đổi hai nhãn cũng không bị kiểm tra hash. Test hồi quy thất bại trước sửa.

**Cách sửa:** 

- Thêm SHA-256 cho nội dung ngữ nghĩa của ánh xạ lớp.
- Chuẩn hóa JSON trước khi băm để kết quả không phụ thuộc LF/CRLF hoặc cách trình bày tệp.
- Từ chối tên lớp trùng nhau trước khi nạp model.

**Tệp liên quan:**

- `src/inference/oe_v2_predictor.py`
- `tests/test_inference_bug_regressions.py`
- `README.md`

### 2. CLI OOD import và thực thi cùng một module hai lần

**Ảnh hưởng:** các lệnh `python -m src.ood.msp`, `python -m src.ood.entropy` và `python -m src.ood.energy` phát `RuntimeWarning` của `runpy`, cảnh báo module đã có trong `sys.modules` trước khi được thực thi. Cách nạp này có thể làm các side effect cấp module chạy hai lần.

**Nguyên nhân gốc:** `src/ood/__init__.py` import nóng các module CLI khi Python mới nạp package `src.ood`; sau đó `python -m` lại thực thi module đích dưới tên `__main__`.

**Bằng chứng tái hiện:** cả ba lệnh `--help` trả mã 0 nhưng đều có `RuntimeWarning` trước sửa. Ba trường hợp hồi quy đều thất bại trước sửa.

**Cách sửa:** chuyển 39 export công khai của package sang lazy import qua `__getattr__`, giữ nguyên tên API công khai nhưng không import trước module CLI.

**Tệp liên quan:**

- `src/ood/__init__.py`
- `tests/test_inference_bug_regressions.py`

## Thay đổi đã thực hiện

- Khóa ánh xạ 120 lớp bằng hash chuẩn hóa `ccd8b9207be5ac03ab2941bdc69a2256cf88fc8fd0acccf64e2cc6100381f8fc`.
- Thêm kiểm tra tên lớp duy nhất.
- Giữ nguyên hash model, hash centroid, kiến trúc model và hai ngưỡng OOD đã khóa.
- Chuyển package OOD sang lazy import mà không đổi danh sách API công khai.
- Thêm test hồi quy cho ánh xạ lớp bị sửa, tên lớp trùng và CLI bị import kép.
- Cập nhật README để mô tả đúng phạm vi kiểm tra toàn vẹn khi khởi động.
- Không hoàn tác hoặc ghi đè các thay đổi chưa commit đã tồn tại trong workspace.

## Kiểm tra đã chạy sau sửa

- `python -m unittest discover -s tests -p "test_*.py" -v`: **120/120 test đạt**.
- `python -m unittest src.test_deployment_smoke src.test_flask_backend src.test_flask_frontend src.test_local_app src.test_ood -v`: **52/52 test đạt**.
- `node --test src/test_frontend_behavior.js`: **48/48 test đạt**.
- Test hồi quy mục tiêu cho hai lỗi mới: **đạt sau khi đã thất bại trên mã trước sửa**.
- `python -m compileall -q app.py src tests`: **mã thoát 0**.
- `python -m pip check`: **không phát hiện dependency hỏng**.
- Kiểm tra 39 public export của `src.ood`: **đạt**.
- Các CLI dữ liệu/OOD được gọi với `--help`: **mã thoát 0**; ba CLI bị lỗi trước đây không còn `RuntimeWarning` do import kép.
- `git diff --check`: **mã thoát 0**; Git chỉ thông báo chính sách chuyển LF sang CRLF của working tree.

Bộ test chính đã nạp model TensorFlow thật và kiểm tra các luồng DOG, UNCERTAIN, NOT_DOG, ảnh OOD, ảnh hỏng, giới hạn upload, WebP và hợp đồng API.

## Vấn đề hoặc giới hạn còn lại

- Không chạy huấn luyện hoặc fine-tuning vì đây là quy trình tốn tài nguyên và có thể ghi đè artifact thử nghiệm.
- Không tái hiệu chỉnh centroid hoặc ngưỡng OOD; các artifact và ngưỡng triển khai hiện tại được giữ nguyên.
- Không triển khai hoặc kiểm thử trên Google Cloud/App Engine.
- TensorFlow báo môi trường Windows hiện tại chạy CPU, cùng một số cảnh báo deprecation/oneDNN; các cảnh báo này không làm test thất bại.
- Một server local đã được khởi chạy trước khi mã được sửa. Server đó cần được khởi động lại để nạp phiên bản mã mới.
- Workspace vẫn có nhiều thay đổi và tệp chưa được theo dõi đã tồn tại từ trước; chúng được bảo toàn và chưa được commit.

