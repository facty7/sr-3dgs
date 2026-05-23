#!/usr/bin/env python3
"""HTTP smoke check for published 3D preview folders.

This does not render WebGL. It catches the common delivery failure where the
viewer works only in theory but assets cannot be loaded over local HTTP.
"""

import argparse
import contextlib
import functools
import http.server
import json
import socketserver
import threading
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@contextlib.contextmanager
def _server(root):
    handler = functools.partial(QuietHandler, directory=str(root))
    with ReusableTCPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=5)


def _fetch(base_url, rel, limit=512_000, method="GET"):
    url = f"{base_url}/{rel.lstrip('/')}"
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        data = b"" if method == "HEAD" else response.read(limit)
        return {
            "url": url,
            "status": response.status,
            "content_type": response.headers.get("content-type", ""),
            "cache_control": response.headers.get("cache-control", ""),
            "content_length": int(response.headers.get("content-length") or 0),
            "bytes_read": len(data),
            "text": data.decode("utf-8", errors="replace") if rel.endswith(".html") or rel.endswith(".json") else "",
        }


def smoke_output(out_dir, root=ROOT):
    out = Path(out_dir).resolve()
    root = Path(root).resolve()
    rel_out = out.relative_to(root)
    problems = []

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    preview = manifest.get("preview") or "preview.html"
    sog = manifest.get("sog") or _first_name(out, "*.sog")
    ply = manifest.get("high_quality_ply") or _first_name(out, "*_high_quality.ply")

    with _server(root) as base_url:
        start = _fetch(base_url, f"{rel_out}/START_HERE.html")
        preview_doc = _fetch(base_url, f"{rel_out}/{preview}")
        diagnostics = _fetch(base_url, f"{rel_out}/diagnostics.json")
        sog_asset = _fetch(base_url, f"{rel_out}/{sog}", method="HEAD")
        ply_asset = _fetch(base_url, f"{rel_out}/{ply}", method="HEAD")

    start_text = start["text"]
    preview_text = preview_doc["text"]
    if "serve_output.py" not in start_text or "file://" not in start_text:
        problems.append("START_HERE.html does not explain local HTTP preview requirements")
    if sog not in preview_text:
        problems.append(f"preview.html does not reference expected SOG asset {sog}")
    if "fetch(" not in preview_text:
        problems.append("preview.html does not appear to fetch a SOG asset")
    if sog_asset["content_length"] <= 0:
        problems.append("SOG asset was not readable over HTTP")
    if ply_asset["content_length"] <= 0:
        problems.append("PLY asset was not readable over HTTP")
    if diagnostics["status"] != 200:
        problems.append("diagnostics.json was not readable over HTTP")
    for label, item in (
        ("START_HERE.html", start),
        ("preview.html", preview_doc),
        ("SOG asset", sog_asset),
        ("PLY asset", ply_asset),
    ):
        cache_control = item.get("cache_control", "")
        if "no-store" not in cache_control:
            problems.append(f"{label} response is missing no-store cache control")

    return {
        "output": str(out),
        "ok": not problems,
        "problems": problems,
        "base_url": start["url"].rsplit("/", 2)[0],
        "checked": {
            "start": start["url"],
            "preview": preview_doc["url"],
            "sog": sog_asset["url"],
            "ply": ply_asset["url"],
            "diagnostics": diagnostics["url"],
        },
    }


def _first_name(out, pattern):
    matches = sorted(Path(out).glob(pattern))
    return matches[0].name if matches else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = smoke_output(args.output_dir)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
