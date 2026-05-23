"""Video frame extraction with quality filtering.

Extracts frames from video, filtering out blurry/low-quality frames
and ensuring adequate scene coverage for COLMAP reconstruction.

Key strategies for good SfM from video:
1. Skip near-duplicate frames (motion < threshold)
2. Detect and skip motion-blurred frames (Laplacian variance)
3. Ensure minimum frame count for COLMAP (30+)
4. Handle various video formats via ffmpeg
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .utils import ensure_dir


def detect_blur_laplacian(img: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
    """Detect blurry images using Laplacian variance.

    Args:
        img: RGB image [H, W, 3], uint8
        threshold: Var below this = blurry

    Returns:
        (is_sharp, variance)
    """
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var >= threshold, lap_var


def frame_difference(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute normalized mean absolute difference between two frames.

    Returns value in [0, 1] — higher means more different.
    """
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32)).mean() / 255.0
    return float(diff)


class VideoFrameExtractor:
    """Extract frames from video with quality and diversity filtering.

    Usage:
        extractor = VideoFrameExtractor("input.mp4", "output_frames/")
        frames = extractor.extract(
            fps=3,                    # Extract ~3 fps
            min_sharpness=100.0,      # Laplacian variance threshold
            min_frame_diff=0.02,      # Skip frames too similar to last kept
            max_frames=300,           # Cap total frames
        )
    """

    @staticmethod
    def _find_ffmpeg():
        """Find ffmpeg binary: system PATH or imageio-ffmpeg bundled."""
        import shutil
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    def __init__(self, video_path: str, output_dir: str):
        self.ffmpeg_bin = self._find_ffmpeg()
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

    def extract(self,
                fps: float = 3.0,
                min_sharpness: float = 100.0,
                min_frame_diff: float = 0.02,
                max_frames: int = 300,
                target_long_edge: int = 1920,
                start_time: Optional[float] = None,
                duration: Optional[float] = None,
                ) -> List[Path]:
        """Extract and filter frames from video.

        Args:
            fps: Target extraction frame rate
            min_sharpness: Minimum Laplacian variance (higher = stricter)
            min_frame_diff: Minimum difference from last kept frame
            max_frames: Maximum frames to extract
            target_long_edge: Resize so long edge = this (0 = no resize)
            start_time: Start time in seconds (None = beginning)
            duration: Duration in seconds (None = entire video)

        Returns:
            List of paths to extracted frames
        """
        ensure_dir(self.output_dir)

        # Get video info
        video_info = self._probe_video()
        print(f"[VideoExtractor] Input: {video_info['width']}x{video_info['height']}, "
              f"{video_info['duration']:.1f}s, {video_info.get('fps', 0):.1f} fps")

        # Step 1: Extract all frames at target fps via ffmpeg
        raw_dir = self.output_dir / "_raw"
        ensure_dir(raw_dir)
        self._ffmpeg_extract(raw_dir, fps, target_long_edge,
                             start_time, duration)

        raw_frames = sorted(raw_dir.glob("*.png")) + sorted(raw_dir.glob("*.jpg"))
        if len(raw_frames) < 10:
            raise RuntimeError(
                f"Only {len(raw_frames)} frames extracted. "
                f"Video too short or ffmpeg failed."
            )
        print(f"[VideoExtractor] Extracted {len(raw_frames)} raw frames at {fps} fps")

        # Step 2: Quality filter (blur detection + diversity)
        import cv2
        kept_paths = []
        last_kept_img = None

        for i, fpath in enumerate(raw_frames):
            if len(kept_paths) >= max_frames:
                break

            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Blur check
            is_sharp, lap_var = detect_blur_laplacian(img_rgb, min_sharpness)
            if not is_sharp:
                continue

            # Diversity check (avoid near-duplicates)
            if last_kept_img is not None:
                diff = frame_difference(last_kept_img, img_rgb)
                if diff < min_frame_diff:
                    continue

            # Keep this frame
            out_path = self.output_dir / f"frame_{len(kept_paths):05d}.png"
            cv2.imwrite(str(out_path), img)
            kept_paths.append(out_path)
            last_kept_img = img_rgb

        # Cleanup raw frames
        shutil.rmtree(raw_dir)

        pct_kept = 100.0 * len(kept_paths) / max(len(raw_frames), 1)
        print(f"[VideoExtractor] Kept {len(kept_paths)}/{len(raw_frames)} "
              f"frames ({pct_kept:.1f}%) after quality filter")

        if len(kept_paths) < 15:
            print("[VideoExtractor] WARNING: Few frames kept. "
                  "Lower min_sharpness or min_frame_diff for better coverage.")

        return kept_paths

    def extract_equirectangular_cubefaces(self,
                                          fps: float = 1.0,
                                          min_sharpness: float = 60.0,
                                          min_frame_diff: float = 0.01,
                                          max_source_frames: int = 80,
                                          face_size: int = 1024,
                                          faces: Tuple[str, ...] = ("front", "right", "back", "left"),
                                          start_time: Optional[float] = None,
                                          duration: Optional[float] = None,
                                          ) -> List[Path]:
        """Extract perspective cube faces from an equirectangular 360 video.

        The generated files are normal pinhole-looking images, which lets the
        existing COLMAP path handle simple 360 object turntable videos.
        """
        raw_dir = self.output_dir / "_raw_equirect"
        ensure_dir(self.output_dir)
        ensure_dir(raw_dir)
        self._ffmpeg_extract(raw_dir, fps, 0, start_time, duration)
        raw_frames = sorted(raw_dir.glob("*.png")) + sorted(raw_dir.glob("*.jpg"))
        if len(raw_frames) < 2:
            raise RuntimeError(f"Only {len(raw_frames)} frames extracted from 360 video.")

        import cv2
        kept = []
        last_kept_img = None
        selected_raw = []
        for fpath in raw_frames:
            if len(selected_raw) >= max_source_frames:
                break
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            is_sharp, _ = detect_blur_laplacian(img_rgb, min_sharpness)
            if not is_sharp:
                continue
            if last_kept_img is not None:
                diff = frame_difference(last_kept_img, img_rgb)
                if diff < min_frame_diff:
                    continue
            selected_raw.append((fpath, img_rgb))
            last_kept_img = img_rgb

        if len(selected_raw) < 2:
            raise RuntimeError("360 extraction kept too few source frames.")

        maps = {face: _build_equirect_face_map(face, face_size) for face in faces}
        for src_index, (fpath, img_rgb) in enumerate(selected_raw):
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            for face in faces:
                map_x, map_y = maps[face]
                face_img = cv2.remap(
                    bgr,
                    map_x * (w - 1),
                    map_y * (h - 1),
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP,
                )
                out_path = self.output_dir / f"frame_{len(kept):05d}_{face}.png"
                cv2.imwrite(str(out_path), face_img)
                kept.append(out_path)

        shutil.rmtree(raw_dir)
        print(
            f"[VideoExtractor] 360 cubemap extraction kept {len(selected_raw)} "
            f"source frames -> {len(kept)} perspective faces"
        )
        return kept

    def _ffmpeg_extract(self, output_dir: Path, fps: float,
                        target_long_edge: int,
                        start_time: Optional[float],
                        duration: Optional[float]):
        """Run ffmpeg to extract frames."""
        vf_parts = [f"fps={fps}"]
        if target_long_edge > 0:
            vf_parts.append(f"scale='min({target_long_edge},iw)':'min({target_long_edge},ih)':force_original_aspect_ratio=decrease")
        vf = ",".join(vf_parts)

        cmd = [
            self.ffmpeg_bin, "-y", "-loglevel", "error",
        ]
        if start_time is not None:
            cmd += ["-ss", str(start_time)]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += [
            "-i", str(self.video_path),
            "-vf", vf,
            "-q:v", "2",  # High quality
            os.path.join(str(output_dir), "frame_%05d.png"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and result.stderr:
            print(f"[VideoExtractor] ffmpeg stderr: {result.stderr[:500]}")

    def _probe_video(self) -> dict:
        """Get video metadata using OpenCV (no ffprobe dependency)."""
        import cv2
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return {"width": 1920, "height": 1080, "duration": 10.0, "fps": 30.0}

        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "duration": 10.0,
        }
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if info["fps"] > 0 and frame_count > 0:
            info["duration"] = frame_count / info["fps"]
        cap.release()
        return info


def _build_equirect_face_map(face: str, face_size: int, fov_degrees: float = 90.0):
    """Return OpenCV remap grids for one horizontal cube face."""
    yaw_by_face = {
        "front": 0.0,
        "right": 90.0,
        "back": 180.0,
        "left": -90.0,
        "up": 0.0,
        "down": 0.0,
    }
    pitch_by_face = {
        "front": 0.0,
        "right": 0.0,
        "back": 0.0,
        "left": 0.0,
        "up": 90.0,
        "down": -90.0,
    }
    if face not in yaw_by_face:
        raise ValueError(f"Unsupported cube face: {face}")

    xs = (np.arange(face_size, dtype=np.float32) + 0.5) / face_size * 2.0 - 1.0
    ys = (np.arange(face_size, dtype=np.float32) + 0.5) / face_size * 2.0 - 1.0
    xx, yy = np.meshgrid(xs, -ys)
    scale = np.tan(np.deg2rad(fov_degrees) * 0.5)
    dirs = np.stack([xx * scale, yy * scale, np.ones_like(xx)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    yaw = np.deg2rad(yaw_by_face[face])
    pitch = np.deg2rad(pitch_by_face[face])
    rot_y = np.array([
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ])
    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch), np.cos(pitch)],
    ])
    dirs = dirs @ (rot_y @ rot_x).T

    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))
    map_x = ((lon / (2.0 * np.pi)) + 0.5).astype(np.float32)
    map_y = (0.5 - lat / np.pi).astype(np.float32)
    return map_x, map_y
