#!/usr/bin/env python3
"""Smoke tests for candidate publishing helpers."""

import tempfile
from pathlib import Path

from publish_clean_candidate import _patch_preview, _write_start_here


def main():
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "preview.html"
        html.write_text(
            """
            const contentUrl = url.searchParams.has('content') ? url.searchParams.get('content') : './toy.sog';
            const sseConfig = {
                contentUrl,
                contents: fetch("toy.sog"),
            };
            """,
            encoding="utf-8",
        )
        patched = _patch_preview(html, "toy_v123.sog")
        assert "toy_v123.sog" in patched, patched
        assert "fetch(\"toy_v123.sog\")" in patched, patched
        assert "./toy_v123.sog" in patched, patched
        assert "toy.sog" not in patched, patched

        out = Path(tmp) / "out"
        out.mkdir()
        _write_start_here(out, "toy", "toy_high_quality.ply", "preview.html", "toy_v123.sog")
        start = (out / "START_HERE.html").read_text(encoding="utf-8")
        assert 'href="toy_v123.sog"' in start, start
        assert 'href="toy.sog"' not in start, start


if __name__ == "__main__":
    main()
