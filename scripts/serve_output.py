#!/usr/bin/env python3
"""Serve published output folders over local HTTP for browser previews."""

import argparse
import functools
import http.server
import socket
import socketserver
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--scene", default="toy")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = root / "output" / args.scene
    start = output_dir / "START_HERE.html"
    if not start.exists():
        raise SystemExit(f"missing published output: {start}")

    handler = functools.partial(NoCacheHandler, directory=str(root))
    try:
        server = ReusableTCPServer((args.host, args.port), handler)
    except OSError as exc:
        if exc.errno in {98, 10048}:
            raise SystemExit(
                f"port {args.port} is already in use; retry with --port {args.port + 1}"
            ) from exc
        raise

    with server:
        url = f"http://{args.host}:{args.port}/output/{args.scene}/START_HERE.html"
        print(f"Serving {root}")
        print(f"Open {url}")
        print("Keep this process running while viewing the 3D preview.")
        server.serve_forever()


if __name__ == "__main__":
    main()
