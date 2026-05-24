"""Step 1: Run COLMAP on original (low-res) images to extract camera poses.

Uses pycolmap's native Python API (pycolmap >= 4.0, COLMAP 4.0+).
Falls back to subprocess if pycolmap unavailable.

CRITICAL: This step MUST use the original unprocessed images.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Sequence

from .utils import ensure_dir


class COLMAPExtractor:
    """Run COLMAP Structure-from-Motion using pycolmap Python API."""

    def __init__(self, image_dir: str, work_dir: str,
                 camera_model: str = "SIMPLE_PINHOLE",
                 colmap_path: str = "colmap",
                 gpu_index: int = 0,
                 camera_model_candidates: Optional[Sequence[str]] = None,
                 min_registered_ratio: float = 0.45,
                 min_registered_images: int = 24):
        self.image_dir = Path(image_dir)
        self.work_dir = Path(work_dir)
        self.sparse_dir = self.work_dir / "sparse" / "0"
        self.database_path = self.work_dir / "database.db"
        self.report_path = self.work_dir / "colmap_report.json"
        self.camera_model = camera_model
        self.camera_model_candidates = tuple(camera_model_candidates or ())
        self.min_registered_ratio = float(min_registered_ratio)
        self.min_registered_images = int(min_registered_images)
        self.colmap_path = colmap_path
        self.gpu_index = gpu_index

    def run(self, force: bool = False):
        """Run full COLMAP pipeline using pycolmap Python API."""
        if self._already_done() and not force:
            print(f"[Step1] COLMAP output exists at {self.sparse_dir}, skipping.")
            if not self.report_path.exists():
                image_count = len(self._image_files())
                report = self._build_report(image_count=image_count, error="")
                self._write_report(report, [report], self._camera_model_sequence())
            return self.sparse_dir

        ensure_dir(self.work_dir)

        image_list = self._image_files()
        if not image_list:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        print(f"[Step1] Found {len(image_list)} images for COLMAP processing.")

        self._run_attempts(image_count=len(image_list))

        self._verify_output()
        return self.sparse_dir

    def _image_files(self):
        image_list = list(self.image_dir.glob("*"))
        return [
            p for p in image_list
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ]

    def _run_attempts(self, image_count: int):
        candidates = self._camera_model_sequence()
        attempts = []
        best_report = None
        best_score = None
        best_sparse = self.work_dir / "_best_sparse"
        last_error = None

        for idx, camera_model in enumerate(candidates, start=1):
            self.camera_model = camera_model
            print(f"[Step1] COLMAP attempt {idx}/{len(candidates)} (camera: {camera_model})")
            self._clear_attempt_outputs()
            try:
                self._run_colmap_once()
                self._verify_output()
                report = self._build_report(image_count=image_count, error="")
                attempts.append(report)
                score = self._report_score(report)
                if best_report is None or score > best_score:
                    best_report = report
                    best_score = score
                    self._copy_sparse(self.sparse_dir, best_sparse)
                if report["meets_quality_target"]:
                    self._write_report(report, attempts, candidates)
                    return
                print(
                    "[Step1] COLMAP attempt below target: "
                    f"{report['registered_images']}/{image_count} images registered "
                    f"({report['registered_ratio']:.2f})."
                )
            except Exception as exc:
                last_error = exc
                report = self._build_report(
                    image_count=image_count,
                    error=f"{type(exc).__name__}: {exc}",
                )
                attempts.append(report)
                print(f"[Step1] COLMAP attempt failed for {camera_model}: {exc}")

        if best_report is not None and best_sparse.exists():
            self.camera_model = best_report["camera_model"]
            self._copy_sparse(best_sparse, self.sparse_dir)
            self._write_report(best_report, attempts, candidates)
            if not best_report["meets_quality_target"]:
                print(
                    "[Step1] WARNING: best COLMAP reconstruction is below target; "
                    "continuing with the best available camera poses."
                )
            return

        failure = {
            "ok": False,
            "image_count": image_count,
            "camera_model": self.camera_model,
            "selected_camera_model": "",
            "registered_images": 0,
            "registered_ratio": 0.0,
            "points3d": 0,
            "meets_quality_target": False,
            "error": str(last_error) if last_error else "all COLMAP attempts failed",
        }
        self._write_report(failure, attempts, candidates)
        if last_error:
            raise last_error
        raise RuntimeError("COLMAP produced no usable reconstruction.")

    def _run_colmap_once(self):
        try:
            self._run_with_pycolmap()
        except ImportError:
            print("[Step1] pycolmap not available, trying system COLMAP...")
            self._run_with_subprocess()

    def _camera_model_sequence(self):
        sequence = [self.camera_model, *self.camera_model_candidates]
        seen = set()
        result = []
        for model in sequence:
            model = str(model).strip().upper()
            if not model or model in seen:
                continue
            seen.add(model)
            result.append(model)
        return result or ["SIMPLE_PINHOLE"]

    def _clear_attempt_outputs(self):
        if self.database_path.exists():
            self.database_path.unlink()
        if self.sparse_dir.parent.exists():
            shutil.rmtree(self.sparse_dir.parent)

    def _copy_sparse(self, src: Path, dst: Path):
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

    def _build_report(self, image_count: int, error: str = ""):
        registered = 0
        points3d = 0
        if not error and self.sparse_dir.exists():
            registered, points3d = self._read_reconstruction_stats()
        ratio = registered / max(1, image_count)
        min_images = min(max(1, self.min_registered_images), max(1, image_count))
        meets_images = registered >= min_images
        meets_ratio = ratio >= self.min_registered_ratio
        return {
            "ok": not bool(error),
            "image_count": int(image_count),
            "camera_model": self.camera_model,
            "selected_camera_model": self.camera_model if not error else "",
            "registered_images": int(registered),
            "registered_ratio": round(float(ratio), 4),
            "points3d": int(points3d),
            "min_registered_images": int(min_images),
            "min_registered_ratio": float(self.min_registered_ratio),
            "meets_min_registered_images": bool(meets_images),
            "meets_min_registered_ratio": bool(meets_ratio),
            "meets_quality_target": bool(not error and meets_images and meets_ratio),
            "error": error,
        }

    def _read_reconstruction_stats(self):
        try:
            import pycolmap
            rec = pycolmap.Reconstruction(str(self.sparse_dir))
            return rec.num_reg_images(), rec.num_points3D()
        except Exception:
            from .utils import read_images_binary, read_points3D_binary
            images = read_images_binary(str(self.sparse_dir / "images.bin"))
            points = read_points3D_binary(str(self.sparse_dir / "points3D.bin"))
            return len(images), len(points)

    @staticmethod
    def _report_score(report):
        return (
            1 if report.get("ok") else 0,
            float(report.get("registered_ratio") or 0.0),
            int(report.get("registered_images") or 0),
            int(report.get("points3d") or 0),
        )

    def _write_report(self, selected_report, attempts, candidates):
        report = dict(selected_report)
        report["selected_camera_model"] = selected_report.get("camera_model", "")
        report["camera_model_candidates"] = list(candidates)
        report["attempts"] = attempts
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            "[Step1] COLMAP report: "
            f"{report['registered_images']}/{report['image_count']} images, "
            f"{report['points3d']} points, camera={report['selected_camera_model']}"
        )

    def _run_with_pycolmap(self):
        """Use pycolmap 4.x Python API for SfM."""
        import pycolmap

        print(f"[Step1] Using pycolmap {pycolmap.__version__} (COLMAP {pycolmap.COLMAP_version})")

        # Step 1a: Import images to database
        self.database_path = self.database_path.absolute()
        if self.database_path.exists():
            self.database_path.unlink()

        reader_opts = pycolmap.ImageReaderOptions()
        reader_opts.camera_model = self.camera_model  # Set camera model name

        # Use absolute paths (pycolmap 4.x requires them)
        db_path = str(self.database_path)
        img_dir = str(self.image_dir.absolute())
        # Create database file (pycolmap requires it to exist before import)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(db_path).touch()
        pycolmap.import_images(
            db_path,
            img_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            options=reader_opts,
        )
        print(f"[Step1] Imported images to database (camera: {self.camera_model}).")

        # Step 1b: Feature extraction
        print("[Step1] Extracting SIFT features...")
        extract_opts = pycolmap.FeatureExtractionOptions()
        extract_opts.use_gpu = True
        extract_opts.gpu_index = str(self.gpu_index)
        extract_opts.max_image_size = 3200

        pycolmap.extract_features(
            db_path,
            img_dir,
            extraction_options=extract_opts,
            device=pycolmap.Device.auto,
        )

        # Step 1c: Feature matching (exhaustive)
        print("[Step1] Matching features (exhaustive)...")
        match_opts = pycolmap.FeatureMatchingOptions()
        match_opts.use_gpu = True
        match_opts.gpu_index = str(self.gpu_index)

        pycolmap.match_exhaustive(
            db_path,
            matching_options=match_opts,
            device=pycolmap.Device.auto,
        )

        # Step 1d: Sparse reconstruction
        print("[Step1] Running incremental mapping...")
        mapper_opts = pycolmap.IncrementalPipelineOptions()

        ensure_dir(self.sparse_dir.parent)
        sparse_root = str(self.sparse_dir.parent.absolute())
        # output_path: where to write reconstruction (it creates sparse/0/, sparse/1/, ...)
        # input_path: empty for fresh reconstruction (no prior model to continue)
        reconstructions = pycolmap.incremental_mapping(
            db_path,
            img_dir,
            sparse_root,
            options=mapper_opts,
            input_path='',
        )

        if not reconstructions:
            raise RuntimeError("COLMAP incremental mapping produced no reconstructions.")

        # Write the largest reconstruction to sparse/0
        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        if self.sparse_dir.exists():
            shutil.rmtree(self.sparse_dir)
        ensure_dir(self.sparse_dir)
        best_rec.write(str(self.sparse_dir))

        print(f"[Step1] Reconstruction complete: {best_rec.num_reg_images()} images, "
              f"{best_rec.num_points3D()} 3D points.")

    def _run_with_subprocess(self):
        """Fallback: use COLMAP binary via subprocess."""
        import subprocess
        print("[Step1] Running COLMAP via subprocess...")
        cmd = [
            self.colmap_path, "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--ImageReader.camera_model", self.camera_model,
            "--SiftExtraction.use_gpu", "1",
            "--SiftExtraction.gpu_index", str(self.gpu_index),
        ]
        subprocess.run(cmd, check=True)

        cmd = [
            self.colmap_path, "exhaustive_matcher",
            "--database_path", str(self.database_path),
            "--SiftMatching.use_gpu", "1",
            "--SiftMatching.gpu_index", str(self.gpu_index),
        ]
        subprocess.run(cmd, check=True)

        ensure_dir(self.sparse_dir)
        cmd = [
            self.colmap_path, "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--output_path", str(self.sparse_dir.parent),
        ]
        subprocess.run(cmd, check=True)

        model_dirs = sorted(
            [d for d in (self.sparse_dir.parent).iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name)
        )
        if not model_dirs:
            raise RuntimeError("COLMAP mapper produced no output models.")
        largest = max(model_dirs, key=lambda d: len(list(d.glob("*.bin"))))
        if largest.name != "0":
            if self.sparse_dir.exists():
                shutil.rmtree(self.sparse_dir)
            largest.rename(self.sparse_dir)
        print(f"[Step1] COLMAP reconstruction done.")

    def _verify_output(self):
        required = ["cameras.bin", "images.bin", "points3D.bin"]
        for fname in required:
            if not (self.sparse_dir / fname).exists():
                raise FileNotFoundError(
                    f"COLMAP output missing {fname} in {self.sparse_dir}"
                )
        print(f"[Step1] COLMAP output verified: {self.sparse_dir}")

    def _already_done(self) -> bool:
        required = ["cameras.bin", "images.bin", "points3D.bin"]
        return all((self.sparse_dir / f).exists() for f in required)
