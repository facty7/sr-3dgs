#!/bin/bash
# ============================================================================
# AutoDL / 云 GPU 一键环境配置脚本
# ============================================================================
# 用法:
#   在 AutoDL 实例上，上传 sr_3dgs 文件夹后执行:
#   bash scripts/setup_autodl.sh
#
# 适配:
#   - AutoDL (autodl.com) 标准镜像: Ubuntu 22.04 + CUDA 12.1 + Miniconda3
#   - 其他云 GPU (Vast.ai / RunPod / 矩池云) 同样适用
# ============================================================================
set -e

echo "========================================"
echo "  SR-3DGS 环境配置"
echo "========================================"

# ── 0. 检测环境 ──
echo "[0/6] 检测环境..."

# 找到 conda
if command -v conda &>/dev/null; then
    CONDA_CMD="conda"
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    CONDA_CMD="conda"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    CONDA_CMD="conda"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
    CONDA_CMD="conda"
else
    echo "ERROR: 找不到 conda，请选择带 Miniconda3 的镜像"
    exit 1
fi

echo "  Conda: $(which conda)"
echo "  Python: $(python3 --version 2>/dev/null || echo 'not found')"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'checking...')"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"

# ── 1. 创建 conda 环境 ──
echo ""
echo "[1/6] 创建 conda 环境 sr_3dgs..."

if $CONDA_CMD env list | grep -q "sr_3dgs"; then
    echo "  环境 sr_3dgs 已存在，跳过创建。"
    echo "  如需重建: conda remove -n sr_3dgs --all"
else
    $CONDA_CMD create -n sr_3dgs python=3.10 -y
    echo "  环境创建完成"
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sr_3dgs

# ── 2. 安装 PyTorch ──
echo ""
echo "[2/6] 安装 PyTorch..."

# 检测 CUDA 版本
CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" 2>/dev/null || echo "12.1")
CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VER | cut -d. -f2)

if [ "$CUDA_MAJOR" -ge 12 ]; then
    TORCH_CUDA="cu121"
elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
    TORCH_CUDA="cu118"
else
    TORCH_CUDA="cu117"
fi

echo "  CUDA Version: $CUDA_VER → PyTorch index: $TORCH_CUDA"

pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" --quiet

python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ── 3. 安装 COLMAP ──
echo ""
echo "[3/6] 安装 COLMAP..."

if command -v colmap &>/dev/null; then
    echo "  COLMAP 已安装: $(colmap --version 2>&1 | head -1 || echo 'OK')"
else
    echo "  尝试 apt 安装..."
    apt-get update -qq && apt-get install -y -qq colmap 2>/dev/null && echo "  COLMAP 安装成功 (apt)" || {
        echo "  apt 不可用，尝试 conda 安装..."
        conda install -c conda-forge colmap -y 2>/dev/null && echo "  COLMAP 安装成功 (conda)" || {
            echo "  WARNING: COLMAP 自动安装失败"
            echo "  请手动安装: https://colmap.github.io/install.html"
            echo "  或使用预编译版本: pip install colmap==3.8"
        }
    }
fi

# ── 4. 安装 Python 依赖 ──
echo ""
echo "[4/6] 安装 Python 依赖..."

# 基础依赖
pip install --quiet \
    numpy'<2.0.0' \
    Pillow \
    scikit-learn \
    opencv-python \
    imageio[ffmpeg] \
    torchmetrics[image] \
    tqdm

# gsplat (3DGS 核心)
echo "  安装 gsplat..."
pip install --quiet gsplat || {
    echo "  gsplat pip 安装失败，尝试从源码安装..."
    pip install --quiet git+https://github.com/nerfstudio-project/gsplat.git || {
        echo "  ERROR: gsplat 安装失败"
        exit 1
    }
}

# Real-ESRGAN (主力超分模型)
echo "  安装 Real-ESRGAN..."
pip install --quiet realesrgan basicsr || {
    echo "  WARNING: Real-ESRGAN 安装失败 (不影响核心流程，但 --sr_model real-esrgan 不可用)"
}

# ffmpeg (视频处理)
if ! command -v ffmpeg &>/dev/null; then
    echo "  安装 ffmpeg..."
    apt-get install -y -qq ffmpeg 2>/dev/null || conda install -c conda-forge ffmpeg -y 2>/dev/null || {
        echo "  WARNING: ffmpeg 安装失败 (视频管线不可用)"
    }
fi

# ── 5. 安装 sr_3dgs ──
echo ""
echo "[5/6] 安装 sr_3dgs..."
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
pip install -e . --quiet
echo "  sr_3dgs 安装完成: $PROJECT_DIR"

# ── 6. 验证 ──
echo ""
echo "[6/6] 验证环境..."

python -c "
import sys
sys.path.insert(0, '.')

# 基础依赖
import numpy;     print(f'  numpy:       {numpy.__version__}')
import torch;     print(f'  torch:       {torch.__version__}')
import PIL;       print(f'  Pillow:      OK')
import cv2;       print(f'  opencv:      {cv2.__version__}')

# gsplat
try:
    import gsplat
    print(f'  gsplat:      {gsplat.__version__}')
except ImportError:
    print('  gsplat:      NOT FOUND (训练不可用)')

# Real-ESRGAN
try:
    import realesrgan
    print('  realesrgan:  OK')
except ImportError:
    print('  realesrgan:  NOT FOUND (可选)')

# sr_3dgs
try:
    import sr_3dgs
    print(f'  sr_3dgs:     {sr_3dgs.__version__}')
except ImportError:
    print('  sr_3dgs:     NOT FOUND')

import subprocess, shutil
if shutil.which('colmap'):
    print('  colmap:      OK')
else:
    print('  colmap:      NOT FOUND (Step1 不可用)')
if shutil.which('ffmpeg'):
    print('  ffmpeg:      OK')
else:
    print('  ffmpeg:      NOT FOUND (视频管线不可用)')
"

echo ""
echo "========================================"
echo "  环境配置完成!"
echo ""
echo "  运行管线:"
echo "    conda activate sr_3dgs"
echo "    python scripts/run_video_pipeline.py --video input.mp4"
echo "========================================"
