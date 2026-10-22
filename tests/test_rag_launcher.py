"""Real socket regressions for the local RAG launcher, without loading models."""

import contextlib
import io
import socket
import subprocess
import sys
import threading
import unittest
import urllib.request
from unittest.mock import patch

import run_rag


class LocalRagLauncherTests(unittest.TestCase):
    def test_occupied_legacy_port_fails_before_index_loading(self):
        with socket.socket() as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            errors = io.StringIO()
            with patch.object(run_rag, "ensure_index_ready") as prepare:
                with contextlib.redirect_stderr(errors):
                    with self.assertRaises(SystemExit) as failure:
                        run_rag.run_server(port, open_browser=False)
            self.assertEqual(failure.exception.code, 1)
            prepare.assert_not_called()
            self.assertIn(f"Không thể mở cổng {port}", errors.getvalue())
            self.assertIn("--port", errors.getvalue())

    def test_socket_is_exclusive_even_against_reuse_address(self):
        with run_rag.create_local_server(0) as server:
            with socket.socket() as competing:
                competing.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                with self.assertRaises(OSError):
                    competing.bind(("127.0.0.1", server.port))
                    competing.listen()

    def test_startup_failure_releases_reserved_port(self):
        with run_rag.create_local_server(0) as reservation:
            port = reservation.port

        def fail_while_reserved():
            with socket.socket() as competing:
                with self.assertRaises(OSError):
                    competing.bind(("127.0.0.1", port))
                    competing.listen()
            raise RuntimeError("simulated index failure")

        with patch.object(run_rag, "ensure_index_ready", side_effect=fail_while_reserved):
            with self.assertRaisesRegex(RuntimeError, "simulated index failure"):
                run_rag.run_server(port, open_browser=False)
        with run_rag.create_local_server(port) as restarted:
            self.assertEqual(restarted.port, port)

    def test_duplicate_cli_is_rejected_and_original_server_still_responds(self):
        def app(environ, start_response):
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"original server"]

        with run_rag.create_local_server(0) as server:
            server.app = app
            duplicate = subprocess.run(
                [sys.executable, "-X", "utf8", str(run_rag.ROOT_DIR / "run_rag.py"),
                 "--port", str(server.port), "--no-browser"],
                cwd=run_rag.ROOT_DIR, capture_output=True, text=True,
                encoding="utf-8", timeout=30,
            )
            self.assertEqual(duplicate.returncode, 1, duplicate.stdout + duplicate.stderr)
            self.assertIn("Không thể mở cổng", duplicate.stderr)
            self.assertNotIn("TensorFlow", duplicate.stdout)
            self.assertNotIn("chỉ mục", duplicate.stdout)
            server.timeout = 5
            worker = threading.Thread(target=server.handle_request, daemon=True)
            worker.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/health", timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.read(), b"original server")
            finally:
                worker.join(timeout=6)
            self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
