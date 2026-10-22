"""Tự động khởi chạy hệ thống RAG cho dự án Dog Breed Classification.

Các chế độ hoạt động:
  1. python run_rag.py           -> Nạp index đã có (tạo nếu thiếu), khởi động Web Server và mở trình duyệt.
  2. python run_rag.py --cli     -> Chế độ hỏi đáp trực tiếp trong Terminal (không cần chờ tải TensorFlow).
  3. python run_rag.py --update  -> Cập nhật các tệp tài liệu thay đổi vào index và thoát.
  4. python run_rag.py --rebuild -> Xây dựng lại toàn bộ index từ đầu và thoát.
  5. python run_rag.py --status  -> Kiểm tra trạng thái index và cấu hình LLM.
"""

import argparse
import os
import socket
import sys
import time
import webbrowser
from pathlib import Path

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))


def ensure_index_ready(force_rebuild: bool = False, update_only: bool = False) -> None:
    """Đảm bảo kho chỉ mục (index) đã sẵn sàng trước khi phục vụ truy vấn."""
    from src.rag.config import load_rag_config
    from src.rag.indexer import Indexer

    config = load_rag_config()
    indexer = Indexer(config)

    index_exists = indexer.chunks_path.exists() and bool(len(indexer.chunks))

    if force_rebuild or not index_exists:
        print("[RAG] 📦 Đang xây dựng kho chỉ mục (index) mới từ các tài liệu...")
        res = indexer.rebuild()
        print(f"[RAG] ✅ Đã lập chỉ mục {res['chunks_count']} đoạn văn từ {res['files_count']} tệp.")
    elif not update_only:
        print(f"[RAG] ⚡ Dùng chỉ mục hiện có ({len(indexer.chunks)} đoạn văn bản).")
        print("[RAG] Khi thêm/sửa tài liệu, chạy run_rag.bat --update để đồng bộ.")
    else:
        print("[RAG] 🔄 Kiểm tra và cập nhật các tệp tài liệu thay đổi (delta update)...")
        res = indexer.update_index()
        if res.get("modified") or res.get("added") or res.get("removed"):
            print(f"[RAG] ✅ Đã cập nhật chỉ mục: {res['chunks_count']} đoạn văn từ {res['files_count']} tệp.")
        else:
            print(f"[RAG] ⚡ Chỉ mục đã ở trạng thái mới nhất ({len(indexer.chunks)} đoạn văn bản).")

    if update_only or force_rebuild:
        return


def show_status() -> None:
    """Hiển thị trạng thái kho tri thức và LLM."""
    from src.rag import get_rag_pipeline

    pipeline = get_rag_pipeline()
    status = pipeline.get_status()
    print("=" * 60)
    print("           TRẠNG THÁI HỆ THỐNG RAG")
    print("=" * 60)
    print(f"Trạng thái hoạt động : {status['status']}")
    print(f"Phiên bản tri thức   : {status['knowledge_version']}")
    print(f"Số lượng tệp         : {status['files_count']}")
    print(f"Vị trí chỉ mục       : {status.get('index_dir', '')}")
    print(f"Số đoạn văn bản      : {status['chunks_count']}")
    print(f"Nhà cung cấp LLM     : {status['llm_provider']}")
    print(f"Trạng thái LLM       : {status['llm_status']}")
    if status.get("missing_config_message"):
        print(f"Lưu ý cấu hình       : {status['missing_config_message']}")
    print("=" * 60)


def run_interactive_cli() -> None:
    """Chế độ hỏi đáp RAG trực tiếp qua dòng lệnh (nhanh chóng, không tải TensorFlow)."""
    ensure_index_ready()
    from src.rag.local_runtime import ensure_local_llm
    from src.rag.config import load_rag_config
    ensure_local_llm(load_rag_config())
    from src.rag import get_rag_pipeline

    pipeline = get_rag_pipeline()
    status = pipeline.get_status()

    print("=" * 65)
    print("  HỆ THỐNG HỎI ĐÁP RAG TIẾNG VIỆT - DOG BREED CLASSIFICATION")
    print("=" * 65)
    print(f"• Phiên bản: {status['knowledge_version']} | {status['chunks_count']} đoạn văn | {status['files_count']} tệp")
    print(f"• Chế độ LLM: {status['llm_provider']} ({status['llm_status']})")
    print("• Gõ 'exit', 'quit' hoặc 'q' để thoát. Gõ 'clear' để xóa lịch sử.")
    print("=" * 65)

    history = []
    while True:
        try:
            query = input("\n[Bạn hỏi]: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Tạm biệt!")
                break
            if query.lower() == "clear":
                history = []
                print("[RAG] Đã xóa lịch sử hội thoại.")
                continue

            print("[RAG] Đang truy xuất tài liệu và sinh câu trả lời...")
            t0 = time.time()
            result = pipeline.query(question=query, history=history)
            latency_ms = (time.time() - t0) * 1000

            print(f"\n[Trả lời ({latency_ms:.1f}ms)]:\n{result['answer']}")

            if result.get("sources"):
                print("\n[Nguồn trích dẫn]:")
                for i, s in enumerate(result["sources"][:5], 1):
                    sec = f" > {s['section']}" if s.get("section") else ""
                    print(f"  {i}. {s['source_path']}{sec}")

            # Lưu vào lịch sử hội thoại
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": result["answer"]})

        except (KeyboardInterrupt, EOFError):
            print("\nTạm biệt!")
            break


def create_local_server(port: int):
    """Reserve the actual serving socket before loading the index or model."""
    from werkzeug.serving import BaseWSGIServer

    class ExclusiveLocalServer(BaseWSGIServer):
        # Windows SO_REUSEADDR can let a second process bind a live listener.
        # Keep the POSIX restart behavior, but require exclusive use on Windows.
        allow_reuse_address = os.name != "nt"

        def server_bind(self):
            try:
                if os.name == "nt":
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                super().server_bind()
            except OSError:
                print(
                    f"[RAG] Không thể mở cổng {port}. Nếu máy chủ đã chạy, mở "
                    f"http://127.0.0.1:{port}/; hoặc chọn cổng khác bằng --port.",
                    file=sys.stderr,
                )
                raise

    # No requests are served until run_server assigns the loaded Flask app.
    return ExclusiveLocalServer("127.0.0.1", port, None)


def run_server(port: int = 5000, open_browser: bool = True) -> None:
    """Hold one socket throughout initialization and serving, without a bind race."""
    with create_local_server(port) as server:
        _serve_initialized_app(server, open_browser)


def _serve_initialized_app(server, open_browser: bool) -> None:
    port = server.port
    ensure_index_ready()
    from src.rag.local_runtime import ensure_local_llm
    from src.rag.config import load_rag_config
    ensure_local_llm(load_rag_config())

    print("\n" + "=" * 65)
    print(f"  KHỞI ĐỘNG MÁY CHỦ FLASK (GIAO DIỆN WEB & RAG)")
    print(f"  Địa chỉ: http://127.0.0.1:{port}/")
    print("=" * 65)
    print("[INFO] Đang nạp TensorFlow và mô hình phân loại...")
    print("[INFO] Vui lòng đợi trong giây lát...\n")

    from app import app
    app.debug = False
    server.app = app

    if open_browser:
        import threading
        import urllib.request

        def _open_when_ready():
            url = f"http://127.0.0.1:{port}/"
            for _ in range(30):
                time.sleep(2)
                try:
                    with urllib.request.urlopen(f"{url}health", timeout=2) as r:
                        if r.status == 200:
                            print(f"[RAG] 🚀 Đang mở trình duyệt: {url}")
                            webbrowser.open(url)
                            break
                except Exception:
                    pass

        t = threading.Thread(target=_open_when_ready, daemon=True)
        t.start()

    server.log_startup()
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tự động khởi chạy hệ thống RAG Dog Breed Classification")
    parser.add_argument("--cli", action="store_true", help="Chạy chế độ hỏi đáp trực tiếp qua Terminal (cực nhanh)")
    parser.add_argument("--rebuild", action="store_true", help="Xây dựng lại chỉ mục từ đầu")
    parser.add_argument("--update", action="store_true", help="Cập nhật tệp mới/sửa đổi vào chỉ mục và thoát")
    parser.add_argument("--status", action="store_true", help="Hiển thị trạng thái chỉ mục và LLM")
    parser.add_argument("--port", type=int, default=5000, help="Cổng chạy máy chủ Web (mặc định 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Không tự động mở trình duyệt")

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.rebuild:
        ensure_index_ready(force_rebuild=True)
    elif args.update:
        ensure_index_ready(update_only=True)
    elif args.cli:
        run_interactive_cli()
    else:
        run_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
