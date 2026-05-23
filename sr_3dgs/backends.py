"""Discovery helpers for optional high-quality reconstruction backends."""

import os
import sys
from pathlib import Path
from typing import Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def candidate_gsplat_paths():
    env = os.environ.get("SR3DGS_GSPLAT_PATH")
    if env:
        yield Path(env)
    root = project_root()
    yield root.parent / "gsplat"
    yield Path.home() / "gsplat"


def ensure_local_gsplat() -> Optional[Path]:
    """Make a sibling/source gsplat checkout importable if pip install is absent."""
    try:
        import gsplat  # noqa: F401
        return None
    except Exception:
        pass

    for path in candidate_gsplat_paths():
        if (path / "gsplat" / "__init__.py").exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            try:
                import gsplat  # noqa: F401
                return path
            except Exception:
                continue
    return None


def backend_status() -> dict:
    status = {"gsplat": False, "gsplat_path": "", "pycolmap": False}
    path = ensure_local_gsplat()
    try:
        import gsplat  # noqa: F401
        status["gsplat"] = True
        status["gsplat_path"] = str(path or "installed")
    except Exception as exc:
        status["gsplat_error"] = str(exc)

    try:
        import pycolmap  # noqa: F401
        status["pycolmap"] = True
    except Exception as exc:
        status["pycolmap_error"] = str(exc)
    return status
