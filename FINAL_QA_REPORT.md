# FINAL QA REPORT

Ngày kiểm tra: 2026-09-25  
Phạm vi: Dataset, Model, OOD artifacts, Threshold config, Predictor, API, Frontend, Tests, Deployment.

## 1. Tóm tắt kiểm tra

| Hạng mục | Kết quả | Evidence chính |
|---|---|---|
| Dataset | INCOMPLETE | Dataset ID có 10.222 ảnh, đủ 120/120 class, 0 ảnh hỏng và split không overlap. Tuy nhiên `data/ood/calibration`, `data/ood/test`, `data/ood_hard` đều rỗng. |
| Model | PASS | Model load thành công; `mobilenet_v2_transfer`, input `(None, 224, 224, 3)`, output `(None, 120)`, 120 class. SHA-256 thực tế khớp manifest: `4695eae55c8464c1a6be3f147828f901259fb2c2541c23c2e36e8b4eb95d5f3d`. |
| OOD artifacts | BLOCKED | `outputs/metrics/ood_benchmark.json` có status `incomplete_missing_ood_data`, `ood_count: 0`; tất cả metric OOD là `null`. Energy calibration có status `missing_ood_calibration_data`. |
| Threshold config | BLOCKED | `ood_config.json` và `ood_thresholds.json` đều ghi `calibrated: false`; ngưỡng đang là `uncertain=0.55`, `not_dog/ood=0.80`, `breed_confidence=0.45`. |
| Predictor | PASS | Predictor load model một lần, kiểm tra shape/probability, dùng MSP và chỉ trả Top-5 ở nhánh `DOG`. |
| API | PASS | `/health`, `/predict` và contract ba trạng thái hoạt động với model thật; smoke/API integration pass. |
| Frontend | PASS | `node src/test_frontend_behavior.js`: 35 pass, 0 fail. UI ẩn breed/Top-5 cho `NOT_DOG` và `UNCERTAIN`. |
| Tests | BLOCKED | Test suite `tests/`: 67 pass, 5 skip vì thiếu OOD fixtures. Suite `src/`: 52 test, 4 fail do các test legacy vẫn bắt synthetic image luôn có breed/Top-5. |
| Deployment | BLOCKED | Import WSGI, startup validation, artifact loading, `/health`, `/predict` smoke đều pass; nhưng package đang ship policy OOD chưa calibrated nên chưa đạt điều kiện production. |

## 2. Xác minh ba behavior

Đã chạy trực tiếp qua Flask `POST /predict` với artifact/model triển khai thật, không mock predictor:

| Case | HTTP | Decision | Kết quả | OOD score | Top-5 |
|---|---:|---|---|---:|---:|
| Dog: `data/test/000621fb3cbb32d8935728e48679680e.jpg` | 200 | `DOG` | `japanese_spaniel`, confidence `0.9595885` | `0.0404115` | 5 |
| Not Dog: ảnh đen đồng nhất | 200 | `NOT_DOG` | `prediction: "not_dog"`, `is_dog: false` | `0.8685973` | 0 |
| Ambiguous: ảnh random noise | 200 | `UNCERTAIN` | `prediction: "uncertain"`, `is_dog: null` | `0.7395250` | 0 |

Ngưỡng runtime tương ứng: `0.55 <= score < 0.80` là vùng uncertain; `score >= 0.80` là not-dog. `/health` trả `ood_loaded: true`, `ood_thresholds_loaded: true`, nhưng `ood_calibrated: false`.

## 3. Test execution evidence

- `python -m unittest discover -s tests -v`: **67 pass, 5 skipped, 0 fail**. Năm skip là các nhóm OOD cần fixture thật nhưng thư mục OOD rỗng.
- `python -m unittest discover -s src -p "test_*.py" -v`: **52 tests, 4 failures**:
  - `src/test_flask_backend.py:94`: synthetic JPEG bắt buộc `prediction` có `index`.
  - `src/test_flask_backend.py:143`: grayscale bắt buộc Top-5 có 5 phần tử.
  - `src/test_flask_backend.py:161`: RGBA PNG bắt buộc Top-5 có 5 phần tử.
  - `src/test_flask_frontend.py:111`: frontend regression bắt synthetic prediction có `index`.

Các failure này phản ánh expectation cũ trước three-state contract: synthetic inputs thực tế có thể trả `UNCERTAIN` và khi đó API chủ ý không trả breed/Top-5. Tuy nhiên test suite trong repo vẫn chưa được cập nhật nên release gate chưa xanh.

- `node src/test_frontend_behavior.js`: **35 pass, 0 fail**.
- `python -m unittest -v tests.test_api_integration src.test_deployment_smoke`: **9 pass, 0 fail**.
- `python -m pytest -q` không chạy được vì environment không cài module `pytest`; repo hiện dùng `unittest` làm runner chính.

## 4. Blockers

### B1 — Chưa có OOD calibration/test data

Evidence:

- `outputs/reports/OOD_DATASET_REPORT.md`: calibration `0`, test `0`, hard `0`.
- `outputs/reports/OOD_BENCHMARK.md`: status `incomplete_missing_ood_data`.
- `outputs/metrics/ood_benchmark.json`: `ood_count: 0`, toàn bộ AUROC/FPR95/AUPR/F1/FAR/FRR là `null`.
- `outputs/metrics/energy_calibration.json`: `status: missing_ood_calibration_data`.

Impact: chưa có bằng chứng định lượng trên representative non-dog và hard-OOD images để xác nhận false acceptance/false rejection của policy.

### B2 — Threshold deployment artifact chưa calibrated

Evidence:

- `outputs/deployment_ready/ood_config.json`: `calibration.calibrated = false`, source `baseline_policy`.
- `outputs/deployment_ready/ood_thresholds.json`: `calibrated = false`, `calibration_date = null`.
- `README.md` ghi rõ phải thay baseline threshold bằng calibrated thresholds trước production use.
- `/health` runtime xác nhận `ood_calibrated: false`.

Impact: behavior mẫu đã chạy đúng theo baseline policy, nhưng chưa đủ cơ sở để gọi policy production-safe.

### B3 — Regression tests chưa đồng bộ với three-state API contract

Evidence: 4 failure trong `src` test discovery như liệt kê ở mục 3. Các test còn yêu cầu mọi synthetic image phải có breed/Top-5, trong khi contract hiện tại quy định `UNCERTAIN`/`NOT_DOG` phải trả `top5: []` và không trả breed candidate.

Impact: full repository QA chưa xanh; cần cập nhật expectation tests hoặc điều chỉnh contract có chủ đích trước khi release.

## 5. Kết luận

BLOCKED
