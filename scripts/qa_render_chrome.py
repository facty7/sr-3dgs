#!/usr/bin/env python3
"""Headless Chrome render QA for published previews.

The script uses an installed Chrome/Edge executable to capture a screenshot,
then performs simple pixel checks so "assets load" and "something rendered"
are both covered. It is intentionally optional: environments without a browser
can skip it while still running the lighter smoke checks.
"""

import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from urllib.request import Request

from PIL import Image, ImageStat


WIN_CHROME_CANDIDATES = [
    r"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    r"/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
]


def _find_browser(explicit=""):
    candidates = []
    if explicit:
        candidates.append(explicit)
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    candidates.extend(WIN_CHROME_CANDIDATES)
    for item in candidates:
        path = Path(item)
        if path.exists() or shutil.which(item):
            return str(path if path.exists() else item)
    return ""


def _screenshot(url, out_png, browser, width, height, wait_ms):
    user_data_dir = Path(tempfile.mkdtemp(prefix="sr3dgs_chrome_"))
    browser_path = str(browser)
    if browser_path.startswith("/mnt/") and shutil.which("powershell.exe") and shutil.which("wslpath"):
        return _screenshot_windows_browser(url, out_png, browser_path, width, height, wait_ms)

    cmd = [
        browser_path,
        "--headless=new",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-dev-shm-usage",
        "--run-all-compositor-stages-before-draw",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        f"--user-data-dir={user_arg}",
        f"--window-size={width},{height}",
        f"--timeout={wait_ms}",
        f"--virtual-time-budget={wait_ms}",
        f"--screenshot={out_arg}",
        url,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    shutil.rmtree(user_data_dir, ignore_errors=True)
    return proc


def _ps_quote(text):
    return "'" + str(text).replace("'", "''") + "'"


class _Ws:
    def __init__(self, url):
        if not url.startswith("ws://"):
            raise ValueError("Only ws:// DevTools URLs are supported")
        host_port, path = url[5:].split("/", 1)
        host, port = host_port.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET /{path} HTTP/1.1\r\n"
            f"Host: {host_port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode("ascii"))
        resp = self.sock.recv(4096)
        if b" 101 " not in resp.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"WebSocket handshake failed: {resp[:120]!r}")

    def send_json(self, payload):
        raw = json.dumps(payload).encode("utf-8")
        header = bytearray([0x81])
        n = len(raw)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header += bytes([0x80 | 126]) + n.to_bytes(2, "big")
        else:
            header += bytes([0x80 | 127]) + n.to_bytes(8, "big")
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(raw))
        self.sock.sendall(bytes(header) + mask + masked)

    def recv_json(self):
        while True:
            first = self.sock.recv(2)
            if len(first) < 2:
                raise RuntimeError("WebSocket closed")
            opcode = first[0] & 0x0F
            length = first[1] & 0x7F
            if length == 126:
                length = int.from_bytes(self.sock.recv(2), "big")
            elif length == 127:
                length = int.from_bytes(self.sock.recv(8), "big")
            masked = bool(first[1] & 0x80)
            mask = self.sock.recv(4) if masked else b""
            data = bytearray()
            while len(data) < length:
                data.extend(self.sock.recv(length - len(data)))
            if masked:
                data = bytearray(b ^ mask[i % 4] for i, b in enumerate(data))
            if opcode == 8:
                raise RuntimeError("WebSocket closed")
            if opcode == 9:
                continue
            if opcode == 1:
                return json.loads(data.decode("utf-8"))

    def close(self):
        self.sock.close()


class _Cdp:
    def __init__(self, ws_url):
        self.ws = _Ws(ws_url)
        self.next_id = 0

    def call(self, method, params=None, timeout=30):
        self.next_id += 1
        msg_id = self.next_id
        self.ws.send_json({"id": msg_id, "method": method, "params": params or {}})
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.ws.recv_json()
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})
        raise TimeoutError(method)

    def close(self):
        self.ws.close()


def _screenshot_windows_browser(url, out_png, browser, width, height, wait_ms):
    browser_win = subprocess.check_output(["wslpath", "-w", browser], text=True).strip()
    temp_win = subprocess.check_output(["cmd.exe", "/C", "echo", "%TEMP%"], text=True).strip()
    stamp = f"{os.getpid()}_{abs(hash(url))}"
    temp_png_win = temp_win.rstrip("\\/") + f"\\sr3dgs_render_{stamp}.png"
    temp_profile_win = temp_win.rstrip("\\/") + f"\\sr3dgs_chrome_profile_{stamp}"
    temp_png_wsl = subprocess.check_output(["wslpath", "-u", temp_png_win], text=True).strip()
    args = [
        "--headless=new",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-dev-shm-usage",
        "--run-all-compositor-stages-before-draw",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        f"--user-data-dir={temp_profile_win}",
        f"--window-size={width},{height}",
        f"--timeout={wait_ms}",
        f"--virtual-time-budget={wait_ms}",
        f"--screenshot={temp_png_win}",
        url,
    ]
    ps_args = " ".join(_ps_quote(arg) for arg in args)
    script = (
        f"& {_ps_quote(browser_win)} {ps_args}; "
        "$code=$LASTEXITCODE; "
        f"Remove-Item -LiteralPath {_ps_quote(temp_profile_win)} -Recurse -Force -ErrorAction SilentlyContinue; "
        "exit $code"
    )
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        text=True,
        capture_output=True,
    )
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if Path(temp_png_wsl).exists():
        shutil.copy2(temp_png_wsl, out_png)
        Path(temp_png_wsl).unlink(missing_ok=True)
    return proc


def _capture_with_cdp(url, out_png, browser, width, height, wait_ms):
    if not browser.startswith("/mnt/") or not shutil.which("powershell.exe") or not shutil.which("wslpath"):
        return None

    browser_win = subprocess.check_output(["wslpath", "-w", browser], text=True).strip()
    temp_win = subprocess.check_output(["cmd.exe", "/C", "echo", "%TEMP%"], text=True).strip().rstrip("\\/")
    stamp = f"{os.getpid()}_{abs(hash(url))}"
    profile_win = temp_win + f"\\sr3dgs_cdp_profile_{stamp}"
    port = 9300 + (os.getpid() % 300)
    args = [
        "--headless=new",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-dev-shm-usage",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_win}",
        f"--window-size={width},{height}",
        "about:blank",
    ]
    ps_args = " ".join(_ps_quote(arg) for arg in args)
    proc = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", f"& {_ps_quote(browser_win)} {ps_args}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        version_url = f"http://127.0.0.1:{port}/json/version"
        for _ in range(80):
            try:
                json.loads(urlopen(version_url, timeout=1).read().decode("utf-8"))
                break
            except Exception:
                time.sleep(0.1)
        new_req = Request(f"http://127.0.0.1:{port}/json/new?{quote(url, safe=':/?&=')}", method="PUT")
        tabs = json.loads(urlopen(new_req, timeout=5).read().decode("utf-8"))
        cdp = _Cdp(tabs["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Emulation.setDeviceMetricsOverride", {
            "width": width,
            "height": height,
            "deviceScaleFactor": 1,
            "mobile": False,
        })
        deadline = time.time() + wait_ms / 1000.0
        state = {}
        while time.time() < deadline:
            time.sleep(0.5)
            state = cdp.call("Runtime.evaluate", {
                "expression": """
(() => {
 const c = document.getElementById('application-canvas');
 const l = document.getElementById('loadingWrap');
 const r = c ? c.getBoundingClientRect() : null;
 return {
   ready: document.readyState,
   canvas: !!c,
   canvasSize: r ? [Math.round(r.width), Math.round(r.height)] : null,
   loadingText: document.getElementById('loadingText')?.textContent || '',
   loadingDisplay: l ? getComputedStyle(l).display : '',
   loadingOpacity: l ? getComputedStyle(l).opacity : '',
   bodyText: document.body.innerText.slice(0, 200)
 };
})()
""",
                "returnByValue": True,
            }).get("result", {}).get("value", {})
            if state.get("canvas") and state.get("loadingDisplay") == "none":
                break
        shot = cdp.call("Page.captureScreenshot", {"format": "png", "fromSurface": True}, timeout=20)
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(out_png).write_bytes(base64.b64decode(shot["data"]))
        cdp.close()
        proc.terminate()
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", f"Remove-Item -LiteralPath {_ps_quote(profile_win)} -Recurse -Force -ErrorAction SilentlyContinue"],
            capture_output=True,
            text=True,
        )
        return {"returncode": 0, "stderr": json.dumps({"dom_state": state})}
    finally:
        if proc.poll() is None:
            proc.terminate()


def _analyze_png(path):
    img = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(img)
    extrema = img.getextrema()
    means = stat.mean
    stddev = stat.stddev
    non_white = 0
    non_black = 0
    sample = img.resize((160, 90))
    for r, g, b in sample.getdata():
        if min(r, g, b) < 245:
            non_white += 1
        if max(r, g, b) > 10:
            non_black += 1
    total = sample.width * sample.height
    return {
        "size": img.size,
        "mean": [round(v, 3) for v in means],
        "stddev": [round(v, 3) for v in stddev],
        "extrema": extrema,
        "non_white_ratio": round(non_white / total, 4),
        "non_black_ratio": round(non_black / total, 4),
        "varied": max(stddev) > 3.0,
    }


def qa_render(url, out_png, browser="", width=1280, height=800, wait_ms=12000):
    browser = _find_browser(browser)
    problems = []
    if not browser:
        return {"ok": False, "skipped": True, "problems": ["Chrome/Edge executable not found"]}

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cdp_result = _capture_with_cdp(url, out_png, browser, width, height, wait_ms)
    if cdp_result is None:
        proc = _screenshot(url, out_png, browser, width, height, wait_ms)
        stderr = proc.stderr
        returncode = proc.returncode
    else:
        stderr = cdp_result.get("stderr", "")
        returncode = cdp_result.get("returncode", 0)
    if returncode != 0:
        problems.append(f"browser exited {returncode}: {stderr[-500:]}")
    if not out_png.exists() or out_png.stat().st_size == 0:
        problems.append("screenshot was not created")
        return {
            "ok": False,
            "skipped": False,
            "browser": browser,
            "url": url,
            "screenshot": str(out_png),
            "problems": problems,
        }

    image = _analyze_png(out_png)
    if not image["varied"]:
        problems.append("screenshot has too little pixel variation")
    if image["non_white_ratio"] < 0.05:
        problems.append("screenshot appears mostly white/blank")
    if image["non_black_ratio"] < 0.05:
        problems.append("screenshot appears mostly black/blank")

    return {
        "ok": not problems,
        "skipped": False,
        "browser": browser,
        "url": url,
        "screenshot": str(out_png),
        "problems": problems,
        "image": image,
        "stderr_tail": stderr[-500:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", default="workspace_video/qa/preview_render.png")
    parser.add_argument("--browser", default="")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--wait_ms", type=int, default=12000)
    parser.add_argument("--allow_skip", action="store_true")
    parser.add_argument(
        "--enable_heavy_browser",
        action="store_true",
        help="Actually launch Chrome/Edge. Required because WebGL screenshot QA can be heavy.",
    )
    args = parser.parse_args()
    if not args.enable_heavy_browser:
        result = {
            "ok": True,
            "skipped": True,
            "reason": "heavy browser render QA requires --enable_heavy_browser",
            "url": args.url,
        }
        print(json.dumps(result, indent=2))
        return
    result = qa_render(args.url, args.out, args.browser, args.width, args.height, args.wait_ms)
    print(json.dumps(result, indent=2))
    if not result["ok"] and not (args.allow_skip and result.get("skipped")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
