"""Heuristics for choosing a super-resolution strategy.

The pipeline keeps COLMAP on original frames for stable geometry. These
heuristics only choose the image set used for training after camera alignment.
They are intentionally conservative because aggressive SR can hallucinate
texture that hurts multi-view consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class SRStrategy:
    mode: str
    model: str
    scale: int
    reason: str
    input_score: Optional[int] = None
    sharpness_p10: Optional[float] = None
    frame_count: Optional[int] = None
    long_edge: Optional[int] = None
    vram_gb: Optional[float] = None
    model_preflight: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def recommend_sr_strategy(
    quality_report: Optional[Dict[str, Any]] = None,
    *,
    preferred_model: str = "real-esrgan",
    preferred_scale: int = 2,
    vram_gb: float = 0.0,
) -> SRStrategy:
    """Recommend a conservative SR mode for general phone captures.

    Returns:
        - off: best default for already-large or weak captures.
        - resize: deterministic x2 upscale for small but not sharp input.
        - model: learned x2 SR for small, sharp, well-covered captures.

    x4 is deliberately not selected automatically for video-to-3DGS because it
    often increases VRAM cost and hallucination risk more than it improves
    geometric reconstruction.
    """
    report = quality_report or {}
    images = report.get("images", {})
    verdict = report.get("verdict", {})
    dims = images.get("dimensions_first_sample") or [0, 0]
    long_edge = int(max(dims) if dims else 0)
    sharpness = images.get("sharpness_laplacian", {})
    sharp_p10 = sharpness.get("p10")
    score = verdict.get("score")
    frame_count = images.get("count")
    problems = set(verdict.get("problems") or [])

    preferred_scale = int(preferred_scale or 2)
    preferred_scale = 2 if preferred_scale <= 1 else min(preferred_scale, 2)

    if long_edge >= 1800:
        return SRStrategy(
            mode="off",
            model=preferred_model,
            scale=1,
            reason="input frames are already near HD/2K; keep geometry-first resolution",
            input_score=score,
            sharpness_p10=sharp_p10,
            frame_count=frame_count,
            long_edge=long_edge,
            vram_gb=vram_gb,
        )

    if "blurry_frames" in problems or (sharp_p10 is not None and sharp_p10 < 45):
        return SRStrategy(
            mode="resize",
            model=preferred_model,
            scale=preferred_scale,
            reason="input is small but too blurry for learned SR; use deterministic resize",
            input_score=score,
            sharpness_p10=sharp_p10,
            frame_count=frame_count,
            long_edge=long_edge,
            vram_gb=vram_gb,
        )

    if frame_count is not None and frame_count < 48:
        return SRStrategy(
            mode="off",
            model=preferred_model,
            scale=1,
            reason="too few views; prioritize capture coverage over SR",
            input_score=score,
            sharpness_p10=sharp_p10,
            frame_count=frame_count,
            long_edge=long_edge,
            vram_gb=vram_gb,
        )

    if 0 < vram_gb < 10:
        return SRStrategy(
            mode="resize" if long_edge < 1400 else "off",
            model=preferred_model,
            scale=preferred_scale if long_edge < 1400 else 1,
            reason="limited VRAM; avoid learned SR training resolution jump",
            input_score=score,
            sharpness_p10=sharp_p10,
            frame_count=frame_count,
            long_edge=long_edge,
            vram_gb=vram_gb,
        )

    if long_edge and long_edge < 1400 and (score is None or score >= 68):
        return SRStrategy(
            mode="model",
            model=preferred_model,
            scale=preferred_scale,
            reason="small, sharp-enough frames with acceptable coverage",
            input_score=score,
            sharpness_p10=sharp_p10,
            frame_count=frame_count,
            long_edge=long_edge,
            vram_gb=vram_gb,
        )

    return SRStrategy(
        mode="off",
        model=preferred_model,
        scale=1,
        reason="default conservative path for general phone video",
        input_score=score,
        sharpness_p10=sharp_p10,
        frame_count=frame_count,
        long_edge=long_edge,
        vram_gb=vram_gb,
    )


def write_strategy(path: str | Path, strategy: SRStrategy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(strategy.to_dict(), indent=2) + "\n", encoding="utf-8")


def adjust_strategy_for_model_preflight(
    strategy: SRStrategy,
    model_preflight: Optional[Dict[str, Any]],
    *,
    allow_download: bool = False,
) -> SRStrategy:
    """Avoid automatic learned SR when weights are unavailable or invalid."""
    if strategy.mode != "model":
        return strategy
    strategy.model_preflight = model_preflight or {}
    if not model_preflight:
        return strategy
    if model_preflight.get("ok") is False:
        strategy.mode = "resize"
        strategy.reason = "learned SR preflight failed; use deterministic resize"
        return strategy
    if model_preflight.get("needs_download") and not allow_download:
        strategy.mode = "resize"
        strategy.reason = (
            "learned SR weights are not local; use deterministic resize "
            "unless downloads are explicitly allowed"
        )
    return strategy
