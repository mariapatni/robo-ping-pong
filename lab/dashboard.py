"""The lab dashboard: everything researched, tried, trained, calibrated and running, on one local page.

    .venv/bin/python -m lab.dashboard            http://localhost:8800
    .venv/bin/python -m lab.dashboard --port 8801

Pages, all read fresh from disk on every request, so anything a script or a
session writes shows up on reload:

  Overview      what is running now, the latest results, recent conclusions
  Research      questions asked and what was concluded     lab/content/research.json
  Experiments   what was tried, and whether it worked      lab/content/experiments.json
  Policies      every keep-up version: its notes (policies/keepup/vN/NOTES.md), scores
                (out/eval_versions/), videos, side by side; render a new one live
  Training      every run's curves: loss and reward from progress.csv (runs from now
                on), performance rebuilt from checkpoints (curve.json) for v1-v7
  Calibration   lens, hand-eye and hover-touch proofs (out/hw/sim_intrinsics, out/handeye,
                out/hover_touch), the real camera fit, the print files
  Live          every job reporting through lab.live (out/runs/<id>/): its description,
                numbers, log and a live view of the sim; start the registered sim
                experiments (lab/experiments.py) from here

Local only: it binds 127.0.0.1, serves files from a fixed set of folders, and
launches only the commands in lab/experiments.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import signal
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUNS = ROOT / "out" / "runs"
SERVE = ("out/", "cad/out/", "policies/keepup/", "lab/content/")  # what /files/ may read
STALE_S = 20.0


# -- reading the disk -------------------------------------------------------------


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def eval_summary(path: Path) -> dict | None:
    """Score a version from its saved evaluation episodes (scripts/eval_keepup.py --json)."""
    eps = read_json(path)
    if not eps:
        return None
    import statistics
    secs = [e["seconds"] for e in eps]
    causes: dict[str, int] = {}
    for e in eps:
        causes[e["cause"]] = causes.get(e["cause"], 0) + 1
    return {"episodes": len(eps), "median_s": statistics.median(secs),
            "reached_60": sum(s >= 60 - 1e-6 for s in secs) / len(eps),
            "mean_bounces": sum(e["bounces"] for e in eps) / len(eps), "causes": causes,
            "seconds": secs}


def progress_csv(path: Path, keep=("time/total_timesteps", "train/loss", "train/value_loss",
                                   "train/policy_gradient_loss", "train/entropy_loss", "train/approx_kl",
                                   "rollout/ep_rew_mean", "rollout/ep_len_mean", "train/learning_rate")) -> dict:
    out: dict[str, list] = {k: [] for k in keep}
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                for k in keep:
                    v = row.get(k, "")
                    out[k].append(float(v) if v not in ("", None) else None)
    except Exception:
        return {}
    return {k: v for k, v in out.items() if any(x is not None for x in v)}


def policies() -> list[dict]:
    base = ROOT / "policies" / "keepup"
    vs = []
    for d in sorted(base.glob("v*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        if not d.is_dir():
            continue
        meta, body = front_matter((d / "NOTES.md").read_text()) if (d / "NOTES.md").exists() else ({}, "")
        videos = sorted({rel(p) for p in (ROOT / "out").glob(f"keepup_{d.name}*.mp4")} |
                        {rel(p) for p in (ROOT / "out" / "dashboard" / "videos").glob(f"{d.name}_*.mp4")})
        vs.append({
            "name": d.name, "meta": meta, "notes": body, "args": read_json(d / "args.json", {}),
            "curriculum": read_json(d / "curriculum.json"),
            "eval": eval_summary(ROOT / "out" / "eval_versions" / f"{d.name}.json"),
            "eval_text": (ROOT / "out" / "eval_versions" / f"{d.name}.txt").read_text()
            if (ROOT / "out" / "eval_versions" / f"{d.name}.txt").exists() else "",
            "videos": videos, "has_best": (d / "best.zip").exists(),
            "checkpoints": len(list((d / "ckpt").glob("keepup_*_steps.zip"))) if (d / "ckpt").exists() else 0,
        })
    return vs


def training() -> list[dict]:
    base = ROOT / "policies" / "keepup"
    runs = []
    for d in sorted(base.glob("v*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        if not d.is_dir():
            continue
        ck = sorted(int(m.group(1)) for p in (d / "ckpt").glob("keepup_*_steps.zip")
                    if (m := re.search(r"keepup_(\d+)_steps", p.name))) if (d / "ckpt").exists() else []
        runs.append({
            "name": d.name, "args": read_json(d / "args.json", {}),
            "steps": [ck[0], ck[-1]] if ck else None,
            "progress": progress_csv(d / "progress.csv") if (d / "progress.csv").exists() else {},
            "curve": read_json(d / "curve.json"), "best": read_json(d / "best.json"),
            "curriculum": read_json(d / "curriculum.json"),
            "live": (d / "progress.csv").exists() and time.time() - (d / "progress.csv").stat().st_mtime < 120,
        })
    return runs


def calibration() -> dict:
    out = ROOT / "out"
    handeye = []
    for d in sorted((out / "handeye").glob("sim_*"), reverse=True):
        r = read_json(d / "results.json")
        if r:
            r.pop("camera", None)
            handeye.append({"dir": rel(d), "name": d.name, "r": r,
                            "views": [rel(p) for p in sorted(d.glob("view*.png"))]})
    hover = []
    for d in sorted((out / "hover_touch").glob("sim_*"), reverse=True):
        r = read_json(d / "results.json")
        if r:
            hover.append({"dir": rel(d), "name": d.name, "summary": r.get("summary", {}),
                          "trials": r.get("trials", []),
                          "shots": [rel(p) for p in sorted(d.glob("spot*_touch.png"))][:8]})
    lens = []
    for d in sorted((out / "hw" / "sim_intrinsics").glob("*"), reverse=True):
        if (d / "report.txt").exists():
            lens.append({"name": d.name, "report": (d / "report.txt").read_text(),
                         "views": rel(d / "views.jpg") if (d / "views.jpg").exists() else None})
    cam = read_json(ROOT / "hw" / "camera.json", {})
    prints = [{"file": "cad/out/paddle_tags.3mf", "title": "Tagged paddle",
               "how": "Hitting face down on textured PEI, no supports. Black body, white inlays.",
               "images": [p for p in ("out/dashboard/prints/paddle_front.jpg", "out/dashboard/prints/paddle_back.jpg",
                                      "out/dashboard/prints/paddle_inlay.jpg", "cad/out/paddle_tags_faces.png")
                          if (ROOT / p).exists()]},
              {"file": "cad/out/charuco_board.3mf", "title": "ChArUco board",
               "how": "Pattern face down on textured PEI. Glue it flat afterwards; measure 7 squares each way.",
               "images": [p for p in ("out/dashboard/prints/board.jpg", "cad/out/charuco_board.png")
                          if (ROOT / p).exists()]}]
    return {"handeye": handeye, "hover": hover, "lens": lens, "camera": {"fit": cam.get("fit"), "f": cam.get("f")},
            "prints": prints}


def live_runs() -> list[dict]:
    runs = []
    now = time.time()
    for d in RUNS.glob("*"):
        s = read_json(d / "status.json")
        if not s:
            continue
        if s.get("state") == "running" and now - s.get("heartbeat", 0) > STALE_S:
            alive = _alive(s.get("pid"))
            s["state"] = "running" if alive and now - s.get("heartbeat", 0) < 600 else "stale"
        s["has_frame"] = (d / "frame.jpg").exists()
        s["has_curves"] = (d / "curves.json").exists()
        runs.append(s)
    runs.sort(key=lambda s: (s["state"] != "running", -s.get("started", 0)))
    return runs


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def overview() -> dict:
    live = live_runs()
    pol = policies()
    best = max((p for p in pol if p["eval"]), key=lambda p: p["eval"]["reached_60"], default=None)
    cal = calibration()
    return {
        "running": [r for r in live if r["state"] == "running"],
        "recent": [r for r in live if r["state"] != "running"][:8],
        "best_policy": best and {"name": best["name"], "reached_60": best["eval"]["reached_60"],
                                 "title": best["meta"].get("title", "")},
        "handeye_latest": cal["handeye"][0] if cal["handeye"] else None,
        "hover_latest": cal["hover"][0] if cal["hover"] else None,
        "research": (read_json(HERE / "content" / "research.json", []) or [])[-4:][::-1],
        "experiments": sorted(read_json(HERE / "content" / "experiments.json", []) or [],
                              key=lambda e: e["date"], reverse=True)[:6],
        "counts": {"research": len(read_json(HERE / "content" / "research.json", []) or []),
                   "experiments": len(read_json(HERE / "content" / "experiments.json", []) or []),
                   "policies": len(pol), "live": sum(r["state"] == "running" for r in live)},
    }


# -- launching ----------------------------------------------------------------------


def launch(name: str, args: dict) -> dict:
    from lab.experiments import EXPERIMENTS
    if name not in EXPERIMENTS:
        raise ValueError(f"no experiment {name!r}")
    ex = EXPERIMENTS[name]
    a = dict(ex["args"])
    for k, v in (args or {}).items():
        if k in a:
            a[k] = int(v)  # integers only: nothing from the page reaches a shell
    cmd = ex["command"](a)
    log = RUNS / f"launch_{time.strftime('%Y%m%d_%H%M%S')}_{name}.log"
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return {"pid": p.pid, "command": " ".join(cmd), "log": rel(log)}


# -- HTTP -----------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        path = unquote(u.path)
        try:
            if path in ("/", "/index.html"):
                return self._send(200, (HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            api = {"/api/overview": overview, "/api/policies": policies, "/api/training": training,
                   "/api/calibration": calibration, "/api/live": live_runs,
                   "/api/research": lambda: read_json(HERE / "content" / "research.json", []),
                   "/api/experiments": lambda: read_json(HERE / "content" / "experiments.json", [])}
            if path in api:
                return self._json(api[path]())
            if path == "/api/registry":
                from lab.experiments import EXPERIMENTS
                return self._json({k: {kk: vv for kk, vv in v.items() if kk != "command"}
                                   for k, v in EXPERIMENTS.items()})
            m = re.match(r"^/live/([\w.\-]+)/(frame\.jpg|stream\.mjpg|curves\.json|log)$", path)
            if m:
                return self._live(m.group(1), m.group(2))
            if path.startswith("/files/"):
                return self._file(path[len("/files/"):])
            self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # keep serving; show what broke
            self._json({"error": repr(e)}, 500)

    do_HEAD = do_GET

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/launch":
                return self._json(launch(body.get("experiment", ""), body.get("args", {})))
            if self.path == "/api/stop":
                s = read_json(RUNS / re.sub(r"[^\w.\-]", "", body.get("id", "")) / "status.json")
                if not s:
                    return self._json({"error": "no such run"}, 404)
                os.killpg(os.getpgid(int(s["pid"])), signal.SIGTERM)
                return self._json({"stopped": s["pid"]})
            self._json({"error": "unknown"}, 404)
        except Exception as e:
            self._json({"error": repr(e)}, 400)

    def _live(self, run_id: str, what: str):
        d = RUNS / run_id
        if what == "frame.jpg":
            p = d / "frame.jpg"
            return self._send(200, p.read_bytes(), "image/jpeg") if p.exists() else self._send(404, b"", "text/plain")
        if what == "curves.json":
            return self._json(read_json(d / "curves.json", {}))
        if what == "stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last, idle = 0.0, 0.0
            while idle < 600:
                p = d / "frame.jpg"
                try:
                    mt = p.stat().st_mtime
                except FileNotFoundError:
                    mt = 0
                if mt and mt != last:
                    data = p.read_bytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    self.wfile.flush()
                    last, idle = mt, 0.0
                time.sleep(0.05)
                idle += 0.05
            return
        self._send(404, b"", "text/plain")

    def _file(self, relpath: str):
        p = (ROOT / relpath).resolve()
        r = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else ""
        if not r or not any(r.startswith(s) for s in SERVE) or "/ckpt/" in r or not p.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if p.suffix in (".txt", ".md", ".log", ".json"):
            ctype += "; charset=utf-8"
        size = p.stat().st_size
        rng = self.headers.get("Range")
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            a = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
            b = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
            b = min(b, size - 1)
            with open(p, "rb") as f:
                f.seek(a)
                data = f.read(b - a + 1)
            return self._send(206, data, ctype, {"Content-Range": f"bytes {a}-{b}/{size}", "Accept-Ranges": "bytes"})
        self._send(200, p.read_bytes(), ctype, {"Accept-Ranges": "bytes"})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=8800)
    a = p.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"lab dashboard on http://localhost:{a.port}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
