"""Setup script for sr_3dgs."""

from setuptools import setup, find_packages

setup(
    name="sr_3dgs",
    version="0.1.0",
    description="Object-focused video/image to 3D Gaussian Splatting delivery pipeline",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="SR 3DGS contributors",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "Pillow>=9.0.0",
        "scikit-learn>=1.0.0",
        "imageio[ffmpeg]>=2.30.0",
        "opencv-python>=4.8.0",
    ],
    extras_require={
        "training": [
            "torch>=2.0.0",
            "torchmetrics[image]>=1.0.0",
            "gsplat>=1.3.0",
        ],
        "sr": ["realesrgan>=0.3.0", "basicsr>=1.4.2"],
        "optional": ["rembg>=2.0.0", "onnxruntime>=1.16.0"],
        "all": [
            "torch>=2.0.0",
            "torchmetrics[image]>=1.0.0",
            "gsplat>=1.3.0",
            "realesrgan>=0.3.0",
            "basicsr>=1.4.2",
            "rembg>=2.0.0",
            "onnxruntime>=1.16.0",
        ],
    },
    scripts=[
        "scripts/run_pipeline.py",
        "scripts/batch_process.py",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
    ],
)
