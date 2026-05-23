"""PlayCanvas SOG export helpers."""

import shutil
import subprocess
from pathlib import Path


def _is_windows_node_cmd(cmd):
    return cmd and str(cmd[0]).replace("\\", "/").startswith("/mnt/")


def _to_cli_path(path, windows_node=False):
    path = Path(path).resolve()
    if windows_node and shutil.which("wslpath"):
        return subprocess.check_output(
            ["wslpath", "-w", str(path)],
            text=True,
        ).strip()
    return str(path)


def _find_splat_transform():
    direct = shutil.which("splat-transform")
    if direct:
        return [direct]
    npx = shutil.which("npx")
    if npx:
        return [npx, "--yes", "@playcanvas/splat-transform"]
    return None


def export_sog(input_ply, output_sog, overwrite=True, iterations=None):
    """Convert a standard 3DGS PLY into PlayCanvas SOG."""
    cmd = _find_splat_transform()
    if not cmd:
        raise RuntimeError(
            "splat-transform not found. Install Node/npm or "
            "`npm install -g @playcanvas/splat-transform`."
        )

    windows_node = _is_windows_node_cmd(cmd)
    input_ply = Path(input_ply)
    output_sog = Path(output_sog)
    output_sog.parent.mkdir(parents=True, exist_ok=True)

    args = list(cmd)
    if overwrite:
        args.append("-w")
    if iterations is not None:
        args += ["--iterations", str(iterations)]
    args += [_to_cli_path(input_ply, windows_node), _to_cli_path(output_sog, windows_node)]
    subprocess.run(args, check=True, cwd=str(Path.home()) if windows_node else None)
    return output_sog


def export_sog_viewer(input_ply, output_html, overwrite=True, unbundled=True, iterations=None):
    """Generate the official PlayCanvas HTML viewer plus SOG assets."""
    cmd = _find_splat_transform()
    if not cmd:
        raise RuntimeError(
            "splat-transform not found. Install Node/npm or "
            "`npm install -g @playcanvas/splat-transform`."
        )

    windows_node = _is_windows_node_cmd(cmd)
    input_ply = Path(input_ply)
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    args = list(cmd)
    if overwrite:
        args.append("-w")
    if iterations is not None:
        args += ["--iterations", str(iterations)]
    args += [_to_cli_path(input_ply, windows_node), _to_cli_path(output_html, windows_node)]
    if unbundled:
        args.append("-U")
    subprocess.run(args, check=True, cwd=str(Path.home()) if windows_node else None)
    return output_html
