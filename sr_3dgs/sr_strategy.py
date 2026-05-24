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
    extraction_coverage_ratio: Optional[float] = None
    extraction_temporal_coverage: Optional[float] = None
    extraction_selected_pass: Optional[str] = None
    sr_risk_flags: Optional[list] = None
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
    extraction = report.get("extraction") or {}
    coverage_ratio = _coverage_ratio(
        extraction.get("selected_count"),
        extraction.get("min_frames"),
    )
    temporal_coverage = _selected_temporal_coverage(extraction)
    temporal_target = _temporal_target(extraction)
    selected_pass = extraction.get("selected_pass")
    risk_flags = _sr_risk_flags(report, problems)

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
            extraction_coverage_ratio=coverage_ratio,
            extraction_temporal_coverage=temporal_coverage,
            extraction_selected_pass=selected_pass,
            sr_risk_flags=risk_flags,
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
            extraction_coverage_ratio=coverage_ratio,
            extraction_temporal_coverage=temporal_coverage,
            extraction_selected_pass=selected_pass,
            sr_risk_flags=risk_flags,
            vram_gb=vram_gb,
        )

    if (
        (frame_count is not None and frame_count < 48)
        or (coverage_ratio is not None and coverage_ratio < 1.0)
        or (temporal_coverage is not None and temporal_coverage < temporal_target)
    ):
        reason = "too few views; prioritize capture coverage over SR"
        if coverage_ratio is not None and coverage_ratio < 1.0:
            reason = "extraction missed its frame coverage target; prioritize capture coverage over SR"
        elif temporal_coverage is not None and temporal_coverage < temporal_target:
            reason = "selected frames cover only part of the video; prioritize full-turn coverage over SR"
        return SRStrategy(
            mode="off",
            model=preferred_model,
            scale=1,
            reason=reason,
            input_score=score,
            sharpness_p10=sharp_p10,
            frame_count=frame_count,
            long_edge=long_edge,
            extraction_coverage_ratio=coverage_ratio,
            extraction_temporal_coverage=temporal_coverage,
            extraction_selected_pass=selected_pass,
            sr_risk_flags=risk_flags,
            vram_gb=vram_gb,
        )

    if risk_flags:
        return SRStrategy(
            mode="resize",
            model=preferred_model,
            scale=preferred_scale,
            reason=(
                "input has multi-view consistency risks; use deterministic "
                "resize instead of learned SR"
            ),
            input_score=score,
            sharpness_p10=sharp_p10,
            frame_count=frame_count,
            long_edge=long_edge,
            extraction_coverage_ratio=coverage_ratio,
            extraction_temporal_coverage=temporal_coverage,
            extraction_selected_pass=selected_pass,
            sr_risk_flags=risk_flags,
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
            extraction_coverage_ratio=coverage_ratio,
            extraction_temporal_coverage=temporal_coverage,
            extraction_selected_pass=selected_pass,
            sr_risk_flags=risk_flags,
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
            extraction_coverage_ratio=coverage_ratio,
            extraction_temporal_coverage=temporal_coverage,
            extraction_selected_pass=selected_pass,
            sr_risk_flags=risk_flags,
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
        extraction_coverage_ratio=coverage_ratio,
        extraction_temporal_coverage=temporal_coverage,
        extraction_selected_pass=selected_pass,
        sr_risk_flags=risk_flags,
        vram_gb=vram_gb,
    )


def _coverage_ratio(selected, target):
    try:
        selected = float(selected)
        target = float(target)
    except (TypeError, ValueError):
        return None
    if target <= 0:
        return None
    return round(selected / target, 3)


def _selected_temporal_coverage(extraction: Dict[str, Any]) -> Optional[float]:
    selected = extraction.get("selected_pass")
    for item in extraction.get("passes") or []:
        if item.get("name") == selected:
            value = item.get("selected_raw_index_coverage")
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _temporal_target(extraction: Dict[str, Any]) -> float:
    try:
        return float(extraction.get("min_span"))
    except (TypeError, ValueError):
        return 0.80


def _sr_risk_flags(report: Dict[str, Any], problems) -> list:
    flags = []
    problem_risks = {
        "near_duplicate_frames",
        "large_view_jumps_or_exposure_changes",
        "bad_exposure_frames",
        "low_contrast_frames",
        "clipped_frames",
    }
    for name in sorted(problem_risks & set(problems or [])):
        flags.append(name)

    images = report.get("images", {})
    frame_diff = images.get("frame_diff") or {}
    frame_health = images.get("frame_health") or {}
    if _lt(frame_diff.get("p50"), 0.015):
        flags.append("near_duplicate_frames")
    if _gt(frame_diff.get("p90"), 0.22):
        flags.append("large_view_jumps_or_exposure_changes")
    if _lt((frame_health.get("contrast") or {}).get("p10"), 5):
        flags.append("low_contrast_frames")
    if _gt((frame_health.get("clipped_ratio") or {}).get("p90"), 0.55):
        flags.append("clipped_frames")

    return sorted(set(flags))


def _lt(value, threshold):
    try:
        return value is not None and float(value) < float(threshold)
    except (TypeError, ValueError):
        return False


def _gt(value, threshold):
    try:
        return value is not None and float(value) > float(threshold)
    except (TypeError, ValueError):
        return False


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
