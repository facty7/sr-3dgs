"""Quality diagnostics and delivery manifest helpers."""

import json
import shutil
import struct
from pathlib import Path

import numpy as np


def _read_splat(path):
    raw = Path(path).read_bytes()
    count = len(raw) // 32
    rows = np.empty((count, 10), dtype=np.float64)
    for i in range(count):
        off = i * 32
        rows[i, :6] = struct.unpack_from("<ffffff", raw, off)
        rows[i, 6:10] = struct.unpack_from("<BBBB", raw, off + 24)
    return rows


def _read_ply(path):
    with open(path, "rb") as f:
        props = []
        count = 0
        while True:
            line = f.readline().decode("utf-8").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float") or line.startswith("property uchar"):
                _, typ, name = line.split()
                props.append((name, {"float": np.float32, "uchar": np.uint8}[typ]))
            elif line == "end_header":
                break
        return np.fromfile(f, dtype=np.dtype(props), count=count)


def diagnose_file(path):
    path = Path(path)
    if path.suffix.lower() == ".splat":
        rows = _read_splat(path)
        xyz = rows[:, :3]
        scale_values = rows[:, 3:6].reshape(-1)
        alpha_values = rows[:, 9]
        scale_kind = "actual"
    elif path.suffix.lower() == ".ply":
        data = _read_ply(path)
        xyz = np.column_stack([data["x"], data["y"], data["z"]])
        scale_values = None
        alpha_values = None
        scale_kind = None
        if "scale_0" in data.dtype.names:
            scales = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]])
            scale_values = scales.reshape(-1)
            scale_kind = "log" if np.nanmedian(scale_values) < -0.25 else "actual"
        if "opacity" in data.dtype.names:
            alpha_values = data["opacity"]
    else:
        raise ValueError(f"Unsupported diagnostics file: {path}")

    center = np.nanmedian(xyz, axis=0)
    radius = np.linalg.norm(xyz - center, axis=1)
    p99 = float(np.percentile(radius, 99))
    max_radius = float(radius.max())
    ok = not (np.isnan(xyz).any() or np.isinf(xyz).any())
    ok = ok and p99 <= 100 and max_radius <= max(100, p99 * 20)

    result = {
        "path": str(path),
        "count": int(len(xyz)),
        "ok": bool(ok),
        "xyz_min": np.nanmin(xyz, axis=0).tolist(),
        "xyz_max": np.nanmax(xyz, axis=0).tolist(),
        "radius_percentiles": {
            "p50": float(np.percentile(radius, 50)),
            "p95": float(np.percentile(radius, 95)),
            "p99": p99,
            "max": max_radius,
        },
    }
    if scale_values is not None:
        if scale_kind == "log":
            comparable_scale_values = np.exp(scale_values)
        else:
            comparable_scale_values = scale_values
        result["scale_kind"] = scale_kind
        result["scale_percentiles"] = {
            "p50": float(np.percentile(scale_values, 50)),
            "p95": float(np.percentile(scale_values, 95)),
            "p99": float(np.percentile(scale_values, 99)),
            "max": float(np.percentile(scale_values, 100)),
        }
        result["scale_actual_percentiles"] = {
            "p50": float(np.percentile(comparable_scale_values, 50)),
            "p95": float(np.percentile(comparable_scale_values, 95)),
            "p99": float(np.percentile(comparable_scale_values, 99)),
            "max": float(np.percentile(comparable_scale_values, 100)),
        }
    if alpha_values is not None:
        result["opacity_percentiles"] = {
            "p50": float(np.percentile(alpha_values, 50)),
            "p95": float(np.percentile(alpha_values, 95)),
            "max": float(np.percentile(alpha_values, 100)),
        }
    return result


def diagnose_paths(paths):
    files = [diagnose_file(p) for p in paths]
    return {
        "ok": all(item["ok"] for item in files),
        "files": files,
    }


def _copy(src, dst):
    src = Path(src)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)
    return ""


def _copy_sog_viewer_assets(viewer_html, dst_dir):
    viewer_html = Path(viewer_html)
    copied = {}
    if not viewer_html.exists():
        return copied

    dst_dir.mkdir(parents=True, exist_ok=True)
    copied["sog_viewer_html"] = _copy(viewer_html, dst_dir / viewer_html.name)

    candidates = [
        viewer_html.with_suffix(".sog"),
        viewer_html.parent / f"{viewer_html.stem}.sog",
        viewer_html.parent / viewer_html.name.replace(".html", ".sog"),
        viewer_html.parent / "index.js",
        viewer_html.parent / "index.css",
        viewer_html.parent / "settings.json",
    ]
    seen = set()
    for path in candidates:
        path = Path(path)
        if path in seen or not path.exists():
            continue
        seen.add(path)
        key = "sog_file" if path.suffix == ".sog" else path.name
        copied[key] = _copy(path, dst_dir / path.name)
    return copied


def _copy_report_pair(results, files, key, dst_dir):
    path = results.get(key)
    if not path:
        return

    src = Path(path)
    copied = _copy(src, dst_dir / src.name)
    if copied:
        files[key] = copied

    html = src.with_suffix(".html")
    copied_html = _copy(html, dst_dir / html.name)
    if copied_html:
        files[f"{key}_html"] = copied_html


def write_delivery_report(delivery_dir, scene_name, results, diagnostics):
    delivery = Path(delivery_dir)
    delivery.mkdir(parents=True, exist_ok=True)

    files = {}
    for key in ("splat_file", "viewer_html"):
        path = results.get(key)
        if path:
            files[key] = _copy(path, delivery / "web" / "splat" / Path(path).name)

    path = results.get("sog_file")
    if path:
        files["sog_file"] = _copy(path, delivery / "web" / "sog" / Path(path).name)

    path = results.get("sog_viewer_html")
    if path:
        files.update(_copy_sog_viewer_assets(path, delivery / "web" / "sog"))

    for key in ("contact_sheet", "turntable"):
        path = results.get(key)
        if path:
            files[key] = _copy(path, delivery / "preview" / Path(path).name)

    path = results.get("standard_ply")
    if path:
        files["standard_ply"] = _copy(path, delivery / "professional" / Path(path).name)
    path = results.get("input_manifest")
    if path:
        files["input_manifest"] = _copy(path, delivery / "reports" / "input_manifest.json")

    reports_dir = delivery / "reports"
    for key in ("sr_manifest", "training_summary", "input_quality_frames", "input_quality_object"):
        _copy_report_pair(results, files, key, reports_dir)

    (delivery / "reports").mkdir(parents=True, exist_ok=True)
    (delivery / "reports" / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    manifest = {
        "scene": scene_name,
        "ok": diagnostics["ok"],
        "files": files,
        "notes": [
            "Use web/sog/*_sog.html for browser/mobile preview.",
            "Use web/sog/*.sog for compact PlayCanvas/SuperSplat-style web delivery.",
            "Use professional/*_standard.ply for high-quality SuperSplat and other 3DGS-aware tools.",
            "Do not deliver if diagnostics ok is false.",
        ],
    }
    (delivery / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (delivery / "README.md").write_text(
        f"""# {scene_name} Delivery

Start here:

- Browser/mobile preview: `web/sog/*_sog.html`
- Compact web asset: `web/sog/*.sog`
- High-quality professional asset: `professional/*_standard.ply`
- Legacy splat fallback: `web/splat/*.splat`
- Diagnostics: `reports/diagnostics.json`
- Input capture quality: `reports/input_quality_*.html`

Only deliver this folder when `manifest.json` has `"ok": true`.
""",
        encoding="utf-8",
    )
    return delivery
