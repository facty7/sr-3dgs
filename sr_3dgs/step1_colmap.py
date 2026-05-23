"""Step 1: Run COLMAP on original (low-res) images to extract camera poses.

Uses pycolmap's native Python API (pycolmap >= 4.0, COLMAP 4.0+).
Falls back to subprocess if pycolmap unavailable.

CRITICAL: This step MUST use the original unprocessed images.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from .utils import ensure_dir


class COLMAPExtractor:
    """Run COLMAP Structure-from-Motion using pycolmap Python API."""

    def __init__(self, image_dir: str, work_dir: str,
                 camera_model: str = "SIMPLE_PINHOLE",
                 colmap_path: str = "colmap",
                 gpu_index: int = 0):
        self.image_dir = Path(image_dir)
        self.work_dir = Path(work_dir)
        self.sparse_dir = self.work_dir / "sparse" / "0"
        self.database_path = self.work_dir / "database.db"
        self.camera_model = camera_model
        self.colmap_path = colmap_path
        self.gpu_index = gpu_index

    def run(self, force: bool = False):
        """Run full COLMAP pipeline using pycolmap Python API."""
        if self._already_done() and not force:
            print(f"[Step1] COLMAP output exists at {self.sparse_dir}, skipping.")
            return self.sparse_dir

        ensure_dir(self.work_dir)

        image_list = list(self.image_dir.glob("*"))
        image_list = [p for p in image_list
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        if not image_list:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        print(f"[Step1] Found {len(image_list)} images for COLMAP processing.")

        try:
            self._run_with_pycolmap()
        except ImportError:
            print("[Step1] pycolmap not available, trying system COLMAP...")
            self._run_with_subprocess()

        self._verify_output()
        return self.sparse_dir

    def _run_with_pycolmap(self):
        """Use pycolmap 4.x Python API for SfM."""
        import pycolmap

        print(f"[Step1] Using pycolmap {pycolmap.__version__} (COLMAP {pycolmap.COLMAP_version})")

        # Map camera model string to pycolmap enum
        camera_model_map = {
            "SIMPLE_PINHOLE": 0,  # CameraModelId.SIMPLE_PINHOLE
            "PINHOLE": 1,
            "SIMPLE_RADIAL": 2,
            "RADIAL": 3,
            "OPENCV": 4,
            "OPENCV_FISHEYE": 5,
            "FULL_OPENCV": 6,
        }
        camera_id = camera_model_map.get(self.camera_model, 0)

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
