"""A live view in the browser: the webcam with whatever a script draws on it.

    viewer = Viewer(port=8765)        # serves http://localhost:8765
    viewer.show(annotated_frame)      # from any thread, as often as frames come

The page is a single <img> fed as an MJPEG stream, so any browser shows it
with no setup. Frames are encoded once and handed to every open page.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>SO-101 live</title>
<style>html,body{margin:0;background:#111;height:100%}img{display:block;width:100%;height:100%;object-fit:contain}</style>
</head><body><img src="/stream" alt="live view"></body></html>"""


class Viewer:
    def __init__(self, port: int = 8765, quality: int = 75):
        self.quality = quality
        self._jpg: bytes | None = None
        self._n = 0
        self._cond = threading.Condition()
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    seen = 0
                    try:
                        while True:
                            with viewer._cond:
                                viewer._cond.wait_for(lambda: viewer._n > seen, timeout=2.0)
                                jpg, seen = viewer._jpg, viewer._n
                            if jpg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.url = f"http://localhost:{port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def show(self, img: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self._cond:
                self._jpg, self._n = buf.tobytes(), self._n + 1
                self._cond.notify_all()

    def close(self) -> None:
        self.server.shutdown()


class Painter:
    """Draws the newest camera frame with a script's state on it, on its own
    thread, so the view keeps moving while the script blocks (an arm move)."""

    def __init__(self, camera, viewer: Viewer, draw, fps: float = 15.0):
        self.camera, self.viewer, self.draw, self.dt = camera, viewer, draw, 1.0 / fps
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        n = 0
        while not self._stop:
            t0 = time.monotonic()
            try:
                frame, _, n = self.camera.read(after=n, timeout=1.0)
                self.viewer.show(self.draw(frame))
            except Exception as e:  # a drawing bug must not stop the session
                print("viewer:", e, flush=True)
            time.sleep(max(0.0, self.dt - (time.monotonic() - t0)))

    def stop(self):
        self._stop = True
