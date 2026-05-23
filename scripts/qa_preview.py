#!/usr/bin/env python3
"""Validate that a published preview page can load its web assets.

This is a lightweight browser-adjacent QA step. It does not require Playwright:
it fetches preview.html through the local server, extracts linked scripts,
stylesheets, settings JSON, and SOG fetch targets, then verifies every asset is
reachable with non-empty content.
"""

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("src"):
            self.assets.append(attrs["src"])
        elif tag == "link" and attrs.get("href") and attrs.get("rel", "").lower() in {"stylesheet", "modulepreload"}:
            self.assets.append(attrs["href"])


def _fetch(url, timeout=10):
    req = Request(url, headers={"User-Agent": "sr-3dgs-preview-qa/1.0"})
    with urlopen(req, timeout=timeout) as res:
        body = res.read()
        return {
            "url": url,
            "status": getattr(res, "status", 200),
            "content_type": res.headers.get("content-type", ""),
            "bytes": len(body),
            "body": body,
        }


def _extract_assets(base_url, html_text):
    parser = AssetParser()
    parser.feed(html_text)
    assets = set(parser.assets)
    for match in re.finditer(r"fetch\(['\"]([^'\"]+)['\"]\)", html_text):
        assets.add(match.group(1))
    for match in re.finditer(r"['\"]([^'\"]*(?:settings\.json|\.sog|index\.css|index\.js))['\"]", html_text):
        assets.add(match.group(1))
    normalized = {urljoin(base_url, asset) for asset in assets if not asset.startswith("data:")}
    return sorted(normalized)


def qa_preview(url):
    problems = []
    fetched = []
    try:
        preview = _fetch(url)
    except Exception as exc:
        return {
            "url": url,
            "ok": False,
            "problems": [f"preview fetch failed: {exc}"],
            "asset_count": 0,
            "fetched": [],
        }
    fetched.append({k: preview[k] for k in ("url", "status", "content_type", "bytes")})
    if preview["status"] >= 400:
        problems.append(f"preview returned HTTP {preview['status']}")
    if preview["bytes"] < 1000:
        problems.append("preview.html is unexpectedly small")

    html = preview["body"].decode("utf-8", errors="replace")
    assets = _extract_assets(url, html)
    if not assets:
        problems.append("no preview assets discovered")

    seen_urls = {url}
    for asset in assets:
        if asset in seen_urls:
            continue
        seen_urls.add(asset)
        try:
            item = _fetch(asset)
            fetched.append({k: item[k] for k in ("url", "status", "content_type", "bytes")})
            if item["status"] >= 400:
                problems.append(f"{asset} returned HTTP {item['status']}")
            if item["bytes"] == 0:
                problems.append(f"{asset} is empty")
        except Exception as exc:
            problems.append(f"{asset} failed: {exc}")

    required_suffixes = [".sog", "index.css", "settings.json"]
    seen_paths = [urlparse(item["url"]).path for item in fetched]
    for suffix in required_suffixes:
        if not any(path.endswith(suffix) for path in seen_paths):
            problems.append(f"missing discovered asset: {suffix}")

    return {
        "url": url,
        "ok": not problems,
        "problems": problems,
        "asset_count": len(fetched) - 1,
        "fetched": fetched,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Preview URL, for example http://127.0.0.1:8765/output/toy/preview.html")
    parser.add_argument("--out", default="", help="Optional JSON report path")
    args = parser.parse_args()
    result = qa_preview(args.url)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
