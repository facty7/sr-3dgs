#!/usr/bin/env python3
"""Smoke tests for the legacy output publisher."""

from publish_output import _patch_viewer_sog


def main():
    html = """
    const contentUrl = './toy.sog';
    const config = {
        contents: fetch("toy.sog"),
        other: fetch('scene.sog'),
    };
    """
    patched = _patch_viewer_sog(html, "toy.sog", "toy_v123.sog")
    assert "./toy_v123.sog" in patched, patched
    assert 'fetch("toy_v123.sog")' in patched, patched
    assert "fetch('toy_v123.sog')" in patched, patched
    assert "toy.sog" not in patched, patched
    assert "scene.sog" not in patched, patched


if __name__ == "__main__":
    main()
