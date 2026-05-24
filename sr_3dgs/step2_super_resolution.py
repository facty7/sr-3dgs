"""Step 2: Apply super-resolution to all input images.

Runs the selected SR model independently on each frame to produce
high-resolution versions. For video-based models (BasicVSR++), processes
frames as a temporal sequence for better consistency.

Input:  Original images from image_dir
Output: Super-resolved images in sr_output_dir (N× original resolution)
"""

import json
import multiprocessing as mp
import os
import queue
import shutil
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .sr_models import get_sr_model
from .sr_models.base import BaseSRModel
from .utils import load_image, save_image, ensure_dir


SR_MODES = {"auto", "off", "copy", "none", "resize", "model"}


class SuperResolutionProcessor:
    """Apply super-resolution to all input images using the selected model."""

    def __init__(self, image_dir: str, output_dir: str,
                 sr_model_name: str = "real-esrgan",
                 scale: int = 4,
                 device: str = "cuda",
                 model_kwargs: Optional[dict] = None,
                 mode: str = "auto",
                 model_load_timeout_s: int = 180,
                 frame_timeout_s: int = 1800,
                 strict_model: bool = False):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.sr_model_name = sr_model_name
        self.scale = scale
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.mode = (mode or "auto").lower()
        self.model_load_timeout_s = int(model_load_timeout_s or 0)
        self.frame_timeout_s = int(frame_timeout_s or 0)
        self.strict_model = bool(strict_model)
        if self.mode not in SR_MODES:
            raise ValueError(
                f"Unknown SR mode '{mode}'. Expected one of: {sorted(SR_MODES)}"
            )
        self._model: Optional[BaseSRModel] = None
        self.manifest_path = self.output_dir / "sr_manifest.json"

    def run(self, force: bool = False):
        """Process all images through the SR model."""
        image_paths = self._get_image_paths()
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        # Check if already done
        effective_mode = self._resolve_mode()

        if self._already_done(image_paths, effective_mode) and not force:
            print(f"[Step2] SR images exist in {self.output_dir}, skipping.")
            return self.output_dir

        ensure_dir(self.output_dir)
        t0 = time.time()

        if effective_mode == "off":
            self._copy_originals(image_paths)
            self._write_manifest(
                image_paths,
                effective_mode=effective_mode,
                status="copied_originals",
                elapsed_s=time.time() - t0,
                model_loaded=False,
            )
            return self.output_dir

        if effective_mode == "resize":
            self._resize_originals(image_paths)
            self._write_manifest(
                image_paths,
                effective_mode=effective_mode,
                status="resized_originals",
                elapsed_s=time.time() - t0,
                model_loaded=False,
            )
            return self.output_dir

        self._clear_stale_outputs(image_paths)
        model_preflight = self._model_preflight()
        if model_preflight:
            self._print_model_preflight(model_preflight)
            if model_preflight.get("ok") is False:
                error = model_preflight.get("error") or "SR model preflight failed"
                print(f"[Step2] SR model preflight failed: {error}")
                if self.strict_model:
                    self._write_manifest(
                        image_paths,
                        effective_mode="model",
                        requested_mode=self.mode,
                        status="model_preflight_failed_strict",
                        elapsed_s=time.time() - t0,
                        model_loaded=False,
                        error=error,
                        model_preflight=model_preflight,
                    )
                    raise RuntimeError(error)
                print("[Step2] Falling back to original resolution (copy images).")
                self._copy_originals(image_paths)
                self._write_manifest(
                    image_paths,
                    effective_mode="off",
                    requested_mode=self.mode,
                    status="model_preflight_failed_copied_originals",
                    elapsed_s=time.time() - t0,
                    model_loaded=False,
                    error=error,
                    model_preflight=model_preflight,
                )
                return self.output_dir
        try:
            worker_result = self._process_model_in_worker(image_paths)
        except Exception as e:
            print(f"[Step2] SR model worker failed: {e}")
            if self.strict_model:
                self._write_manifest(
                    image_paths,
                    effective_mode="model",
                    requested_mode=self.mode,
                    status="model_worker_failed_strict",
                    elapsed_s=time.time() - t0,
                    model_loaded=False,
                    error=str(e),
                    model_preflight=model_preflight,
                )
                raise
            print("[Step2] Falling back to original resolution (copy images).")
            self._copy_originals(image_paths)
            self._write_manifest(
                image_paths,
                effective_mode="off",
                requested_mode=self.mode,
                status="model_worker_failed_copied_originals",
                elapsed_s=time.time() - t0,
                model_loaded=False,
                error=str(e),
                model_preflight=model_preflight,
            )
            return self.output_dir

        print(f"[Step2] Super-resolution complete. Output: {self.output_dir}")
        self._write_manifest(
            image_paths,
            effective_mode="model",
            status="model_processed",
            elapsed_s=time.time() - t0,
            model_loaded=True,
            model_name=worker_result.get("model_name", self.sr_model_name),
            model_preflight=model_preflight,
        )
        return self.output_dir

    def _model_preflight(self) -> dict:
        try:
            model = get_sr_model(
                self.sr_model_name,
                scale=self.scale,
                device=self.device,
                **self.model_kwargs,
            )
            describe = getattr(model, "describe_weights", None)
            if callable(describe):
                return describe()
            return {
                "ok": True,
                "model": model.name,
                "requested_scale": int(self.scale),
                "needs_download": None,
                "weights_exist": None,
            }
        except Exception as exc:
            return {
                "ok": False,
                "model": self.sr_model_name,
                "requested_scale": int(self.scale),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _print_model_preflight(self, preflight: dict):
        status = "ok" if preflight.get("ok", True) else "warning"
        print(
            "[Step2] SR model preflight "
            f"({status}): {preflight.get('model', self.sr_model_name)}"
        )
        if preflight.get("needs_download"):
            print(
                "[Step2] SR weights are not local; model load may download "
                f"{preflight.get('weight_name', 'weights')}"
            )
        if preflight.get("explicit_exists") is False:
            print(
                "[Step2] WARNING: explicit SR model path is missing: "
                f"{preflight.get('explicit_model_path')}"
            )

    def _process_model_in_worker(self, image_paths: List[Path]) -> dict:
        """Run learned SR in a child process so load/inference can time out."""
        if self.model_load_timeout_s <= 0 and self.frame_timeout_s <= 0:
            model = self._get_model()
            if model.requires_temporal_input:
                self._process_temporal(image_paths, model)
            else:
                self._process_per_image(image_paths, model)
            return {"model_name": model.name}

        ctx = _mp_context()
        result_q = ctx.Queue()
        proc = ctx.Process(
            target=_model_worker,
            args=(
                [str(p) for p in image_paths],
                str(self.output_dir),
                self.sr_model_name,
                int(self.scale),
                self.device,
                dict(self.model_kwargs),
                result_q,
            ),
        )
        proc.start()
        self._wait_for_worker(proc, image_paths)

        try:
            result = result_q.get(timeout=2)
        except queue.Empty as exc:
            raise RuntimeError(
                f"SR model worker exited without a result (code {proc.exitcode})"
            ) from exc
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "SR model worker failed")
        missing = [p.name for p in image_paths if not (self.output_dir / p.name).exists()]
        if missing:
            raise RuntimeError(
                "SR model worker finished but did not produce all images: "
                + ", ".join(missing[:5])
            )
        return result

    def _wait_for_worker(self, proc, image_paths: List[Path]):
        start = time.time()
        last_progress_time = start
        last_done = 0
        expected_names = {p.name for p in image_paths}
        while proc.is_alive():
            proc.join(1)
            if not proc.is_alive():
                break
            done = sum(1 for p in self.output_dir.iterdir()
                       if p.name in expected_names and p.is_file())
            if done > last_done:
                last_done = done
                last_progress_time = time.time()
            now = time.time()
            if last_done == 0 and self.model_load_timeout_s > 0:
                if now - start > self.model_load_timeout_s:
                    self._stop_worker(proc)
                    raise TimeoutError(
                        "SR model worker produced no images before "
                        f"load timeout ({self.model_load_timeout_s}s)"
                    )
            elif last_done < len(image_paths) and self.frame_timeout_s > 0:
                if now - last_progress_time > self.frame_timeout_s:
                    self._stop_worker(proc)
                    raise TimeoutError(
                        "SR model worker made no image progress for "
                        f"{self.frame_timeout_s}s ({last_done}/{len(image_paths)} done)"
                    )

    @staticmethod
    def _stop_worker(proc):
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)

    def _process_per_image(self, image_paths: List[Path], model: BaseSRModel):
        """Process images independently (Real-ESRGAN, DAT, SUPIR)."""
        total = len(image_paths)
        for i, img_path in enumerate(image_paths):
            out_path = self.output_dir / img_path.name
            if out_path.exists():
                print(f"  [{i+1}/{total}] Skipping (exists): {img_path.name}")
                continue
            print(f"  [{i+1}/{total}] Processing: {img_path.name}")
            img = load_image(str(img_path))
            sr_img = model.process_image(img)
            save_image(str(out_path), sr_img)

    def _process_temporal(self, image_paths: List[Path], model: BaseSRModel):
        """Process images as a temporal sequence (BasicVSR++)."""
        print(f"[Step2] Loading {len(image_paths)} frames for temporal SR...")
        images = [load_image(str(p)) for p in image_paths]

        print(f"[Step2] Running temporal super-resolution with {model.name}...")
        sr_images = model.process_batch(images)

        for i, img_path in enumerate(image_paths):
            out_path = self.output_dir / img_path.name
            save_image(str(out_path), sr_images[i])

    def _get_model(self) -> BaseSRModel:
        if self._model is None:
            self._model = get_sr_model(
                self.sr_model_name,
                scale=self.scale,
                device=self.device,
                **self.model_kwargs,
            )
            self._model.load()
            print(f"[Step2] Loaded SR model: {self._model.name}")
        return self._model

    def _resolve_mode(self) -> str:
        if self.mode in {"off", "copy", "none"}:
            return "off"
        if self.mode == "resize":
            return "resize" if self.scale > 1 else "off"
        if self.mode == "model":
            return "model" if self.scale > 1 else "off"
        if self.scale <= 1:
            return "off"
        if self.sr_model_name.lower() in {"off", "none", "copy"}:
            return "off"
        if self.sr_model_name.lower() in {"resize", "bicubic", "lanczos"}:
            return "resize"
        return "model"

    def _get_image_paths(self) -> List[Path]:
        paths = sorted([
            p for p in self.image_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ])
        return paths

    def _already_done(self, image_paths: List[Path],
                      expected_mode: Optional[str] = None) -> bool:
        """Check if all output images already exist."""
        if not self.output_dir.exists():
            return False
        for p in image_paths:
            if not (self.output_dir / p.name).exists():
                return False
        if expected_mode and not self.manifest_path.exists():
            print("[Step2] Existing SR output has no manifest; regenerating.")
            return False
        if expected_mode and self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                return False
            if manifest.get("effective_mode") != expected_mode:
                print(
                    "[Step2] Existing SR output mode "
                    f"({manifest.get('effective_mode')}) does not match requested "
                    f"mode ({expected_mode}); regenerating."
                )
                return False
            if int(manifest.get("scale", self.scale)) != int(self.scale):
                print(
                    "[Step2] Existing SR output scale "
                    f"({manifest.get('scale')}) does not match requested scale "
                    f"({self.scale}); regenerating."
                )
                return False
            if expected_mode == "model" and manifest.get("sr_model") != self.sr_model_name:
                print(
                    "[Step2] Existing SR model "
                    f"({manifest.get('sr_model')}) does not match requested "
                    f"model ({self.sr_model_name}); regenerating."
                )
                return False
        return True

    def _copy_originals(self, image_paths):
        """Copy original images to output when SR is unavailable."""
        self._clear_stale_outputs(image_paths)
        for p in image_paths:
            dst = self.output_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
        print(f"[Step2] Copied {len(image_paths)} images at original resolution.")

    def _resize_originals(self, image_paths):
        """Deterministic non-learned upscale for ablation and low-risk runs."""
        self._clear_stale_outputs(image_paths)
        for p in image_paths:
            dst = self.output_dir / p.name
            if dst.exists():
                continue
            with Image.open(p) as img:
                img = img.convert("RGB")
                w, h = img.size
                resized = img.resize(
                    (max(1, int(round(w * self.scale))),
                     max(1, int(round(h * self.scale)))),
                    Image.Resampling.LANCZOS,
                )
                resized.save(dst)
        print(f"[Step2] Resized {len(image_paths)} images with Lanczos x{self.scale}.")

    def _clear_stale_outputs(self, image_paths):
        valid_names = {p.name for p in image_paths}
        if not self.output_dir.exists():
            return
        for p in self.output_dir.iterdir():
            if p.name == self.manifest_path.name:
                continue
            if p.is_file() and (
                p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                or p.name not in valid_names
            ):
                p.unlink()

    def _write_manifest(self, image_paths, effective_mode: str,
                        status: str,
                        elapsed_s: float = 0.0,
                        model_loaded: bool = False,
                        model_name: str = "",
                        requested_mode: Optional[str] = None,
                        error: str = "",
                        model_preflight: Optional[dict] = None):
        sample = image_paths[0]
        with Image.open(sample) as img:
            input_w, input_h = img.size
        output_sample = self.output_dir / sample.name
        if output_sample.exists():
            with Image.open(output_sample) as img:
                output_w, output_h = img.size
        else:
            output_w, output_h = input_w, input_h

        manifest = {
            "requested_mode": requested_mode or self.mode,
            "effective_mode": effective_mode,
            "status": status,
            "sr_model": self.sr_model_name,
            "model_name": model_name,
            "scale": self.scale,
            "device": self.device,
            "model_loaded": model_loaded,
            "input_dir": str(self.image_dir),
            "output_dir": str(self.output_dir),
            "image_count": len(image_paths),
            "input_size": [input_w, input_h],
            "output_size": [output_w, output_h],
            "effective_scale": [
                output_w / max(input_w, 1),
                output_h / max(input_h, 1),
            ],
            "elapsed_s": round(float(elapsed_s), 3),
        }
        if model_preflight:
            manifest["model_preflight"] = model_preflight
        if error:
            manifest["error"] = error
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[Step2] Wrote SR manifest: {self.manifest_path}")

    def cleanup(self):
        if self._model is not None:
            self._model.unload()


def _mp_context():
    try:
        return mp.get_context("spawn")
    except ValueError:
        return mp.get_context()


def _model_worker(image_paths, output_dir, sr_model_name, scale, device,
                  model_kwargs, result_q):
    try:
        model = get_sr_model(
            sr_model_name,
            scale=scale,
            device=device,
            **(model_kwargs or {}),
        )
        model.load()
        out_dir = Path(output_dir)
        paths = [Path(p) for p in image_paths]
        if model.requires_temporal_input:
            images = [load_image(str(p)) for p in paths]
            sr_images = model.process_batch(images)
            for img_path, sr_img in zip(paths, sr_images):
                save_image(str(out_dir / img_path.name), sr_img)
        else:
            total = len(paths)
            for idx, img_path in enumerate(paths, start=1):
                print(f"  [{idx}/{total}] Processing: {img_path.name}", flush=True)
                img = load_image(str(img_path))
                sr_img = model.process_image(img)
                save_image(str(out_dir / img_path.name), sr_img)
        result_q.put({"ok": True, "model_name": model.name})
        model.unload()
    except Exception as exc:
        result_q.put({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
