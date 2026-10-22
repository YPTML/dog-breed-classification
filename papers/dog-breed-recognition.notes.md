# Ghi chú đối chiếu hình trong paper Dog Breed Recognition

Nguồn: `papers/dog-breed-recognition.pdf`, bài *An Intelligent Dog Breed Recognition
System Using Deep Learning*, E. S. Madhan, Arihant Kaushik, Rohit Raju (2022).
Đây là ghi chú đọc trực tiếp hình đã render, không phải kết quả OCR tự động hoặc
kết quả huấn luyện của dự án hiện tại. Số trang dưới đây là trang PDF từ 1 đến 14.

## Trang PDF 9 — hình 12: kiến trúc trong ảnh chụp mã

Ảnh mã cho thấy Sequential theo thứ tự: data_augmentation, base_model,
Dropout(0.5), Dropout(0.2), Dense(OUTPUT_SHAPE, activation='softmax').
Hình không tự chứng minh kiến trúc OE V2 trong dự án hiện tại dùng cùng các lớp này.

## Trang PDF 10 — hình 13: cấu hình huấn luyện

Ảnh mã dùng Adam, loss categorical_crossentropy, metric accuracy; epochs đặt tối đa
100. EarlyStopping theo val_accuracy với patience=3. model.fit nhận train_data,
validation_data=test_data và callback early_stopping. Tên biến test_data trong hình
không chứng minh có một tập kiểm thử độc lập sau bước chọn mô hình.

## Trang PDF 10–11 — hình 14–16: accuracy và loss

Đường training accuracy tăng cao hơn validation accuracy. Chú thích hình 14 và
phần Results trang PDF 11 báo cáo training accuracy khoảng 90%, validation accuracy
81%. Không biến 81% này thành độ chính xác kiểm thử độc lập của mô hình dự án.

## Trang PDF 12 — bảng 1 và bảng 2

Bảng 1 báo cáo Basic CNN: 64%, 60 phút; model kết hợp MobileNetV2: 77%, 90 phút;
fine-tuned model dùng Adam và categorical cross entropy: 81%, 60 phút.
Bảng 2 ghi bộ nhớ gốc 1.2 GB và sau tối ưu là 965, nhưng ô 965 không ghi đơn vị.
Không tự thêm đơn vị vào con số đó nếu chưa có nguồn khác xác nhận.

## Giới hạn của việc đọc PDF

Chỉ mục PDF tự động đọc lớp chữ và giữ số trang. Ghi chú này bổ sung các hình đã
kiểm tra ở trang 9–12; các hình khác không được coi là đã diễn giải toàn bộ.
Trang PDF 3 có chuỗi kích thước 224 * 244 * 3 trong văn bản; cần giữ nguyên khi
trích dẫn paper và đối chiếu với cấu hình 224×224 của dự án trước khi sử dụng.
