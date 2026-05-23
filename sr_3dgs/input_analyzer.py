"""Input media inspection for choosing a reconstruction path."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class VideoInputInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    projection: str
    reconstruction_mode: str
    reason: str

    def to_dict(self):
        return asdict(self)


def analyze_video(video_path: str, projection: str = "auto") -> VideoInputInfo:
    """Inspect a video and classify it as perspective or 360 equirectangular."""
    import cv2

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    cap.release()

    aspect = width / max(height, 1)
    requested = projection.lower()
    if requested not in {"auto", "perspective", "equirectangular"}:
        raise ValueError("projection must be auto, perspective, or equirectangular")

    if requested == "equirectangular":
        inferred_projection = "equirectangular"
        reason = "user override"
    elif requested == "perspective":
        inferred_projection = "perspective"
        reason = "user override"
    elif 1.85 <= aspect <= 2.15:
        inferred_projection = "equirectangular"
        reason = f"auto-detected 2:1-ish aspect ratio ({aspect:.2f})"
    else:
        inferred_projection = "perspective"
        reason = f"auto-detected non-2:1 aspect ratio ({aspect:.2f})"

    mode = "cubemap_frames" if inferred_projection == "equirectangular" else "perspective_frames"
    return VideoInputInfo(
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frame_count=frame_count,
        projection=inferred_projection,
        reconstruction_mode=mode,
        reason=reason,
    )
