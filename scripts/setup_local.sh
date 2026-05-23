#!/bin/bash
# ============================================================================
# SR-3DGS 本地环境配置 (WSL / Linux)
# ============================================================================
# 用法:
#   bash scripts/setup_local.sh                  # 自动找现有环境
#   bash scripts/setup_local.sh gs_dev           # 安装到已有环境
#   bash scripts/setup_local.sh sr_3dgs --new    # 强制创建新环境
# ============================================================================
set -e

ENV_NAME="${1:-}"
FORCE_NEW="${2:-}"

echo "========================================"
echo "  SR-3DGS 本地环境配置"
echo "========================================"

# ── 1. 找 conda ──
CONDA_BASE=""
CONDA_SH=""

for candidate in \
    "$HOME/yes" \
    "$HOME/miniconda3" \
    "$HOME/anaconda3" \
    "$HOME/mambaforge" \
    "/opt/conda" \
    "/root/miniconda3" \
    "$(conda info --base 2>/dev/null || true)" \
; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$candidate"
        CONDA_SH="$candidate/etc/profile.d/conda.sh"
        break
    fi
done

if [ -z "$CONDA_SH" ]; then
    echo "ERROR: 找不到 conda"
    exit 1
fi

echo "  Conda: $CONDA_BASE"
source "$CONDA_SH"

# ── 2. 检测 PyPI 连通性 → 选择镜像 ──
echo ""
echo "检测 PyPI 连通性..."

MIRROR=""
if curl -s --connect-timeout 3 https://pypi.org/simple/ >/dev/null 2>&1; then
    echo "  PyPI 直连 OK"
else
    # 测国内镜像
    for mirror in \
        "https://pypi.tuna.tsinghua.edu.cn/simple" \
        "https://mirrors.aliyun.com/pypi/simple/" \
        "https://mirror.sjtu.edu.cn/pypi/web/simple/" \
        "https://pypi.doubanio.com/simple/"; do
        if curl -s --connect-timeout 3 "$mirror" >/dev/null 2>&1; then
            MIRROR="$mirror"
            echo "  使用镜像: $MIRROR"
            break
        fi
    done
    if [ -z "$MIRROR" ]; then
        echo "  WARNING: 所有镜像不可达，pip 可能失败"
    fi
fi

if [ -n "$MIRROR" ]; then
    PIP_INSTALL="pip install --quiet -i $MIRROR --trusted-host $(echo $MIRROR | cut -d/ -f3)"
else
    PIP_INSTALL="pip install --quiet"
fi

# ── 3. 确定环境 ──
if [ -z "$ENV_NAME" ]; then
    EXISTING=$(conda env list 2>/dev/null | grep -oP '^\S+' | grep -E 'gs_dev|gs_env|gsplat' | head -1 || true)
    if [ -n "$EXISTING" ]; then
        echo ""
        echo "  检测到已有环境: $EXISTING (包含 gsplat)"
        echo "  按 Enter 直接安装到 $EXISTING, 或输入新名字创建新环境:"
        read -r USER_CHOICE
        if [ -z "$USER_CHOICE" ]; then
            ENV_NAME="$EXISTING"
        else
            ENV_NAME="$USER_CHOICE"
        fi
    else
        ENV_NAME="sr_3dgs"
    fi
fi

if [ "$FORCE_NEW" = "--new" ]; then
    conda create -n "$ENV_NAME" python=3.10 -y
fi

if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "  使用环境: $ENV_NAME (已存在)"
else
    echo "  创建环境: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"
echo "  Python: $(python --version 2>&1)"

# ── 4. 安装依赖 ──
echo ""
echo "安装依赖..."

install_if_missing() {
    python -c "import $1" 2>/dev/null && echo "  [skip] $1" || {
        echo "  [install] $2"
        $PIP_INSTALL $2 || echo "  [WARN] $2 安装失败，尝试继续..."
    }
}

# PyTorch
python -c "import torch; print('  [skip] torch', torch.__version__)" 2>/dev/null || {
    echo "  安装 PyTorch..."
    if [ -n "$MIRROR" ]; then
        pip install --quiet torch torchvision
    else
        pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
    fi
}

# 基础包
for pair in \
    "numpy:numpy<2.0.0" \
    "PIL:Pillow" \
    "sklearn:scikit-learn" \
    "cv2:opencv-python" \
    "imageio:imageio[ffmpeg]" \
    "torchmetrics:torchmetrics[image]" \
    "tqdm:tqdm" \
; do
    mod="${pair%%:*}"
    pkg="${pair##*:}"
    install_if_missing "$mod" "$pkg"
done

# gsplat
python -c "import gsplat" 2>/dev/null && echo "  [skip] gsplat" || {
    echo "  安装 gsplat..."
    $PIP_INSTALL gsplat 2>/dev/null || {
        echo "  pip 失败，从 GitHub 安装..."
        pip install --quiet git+https://github.com/nerfstudio-project/gsplat.git
    }
}

# realesrgan
python -c "import realesrgan" 2>/dev/null && echo "  [skip] realesrgan" || {
    echo "  安装 Real-ESRGAN..."
    $PIP_INSTALL realesrgan basicsr 2>/dev/null && echo "  Real-ESRGAN OK" || {
        echo "  镜像安装失败，尝试 GitHub 源码..."
        pip install --quiet git+https://github.com/xinntao/Real-ESRGAN.git 2>/dev/null || {
            echo "  [WARN] Real-ESRGAN 安装失败 (网络问题)"
            echo "  不影响核心流程，后续可手动装"
            echo "  或使用: --sr_scale 2 (超分2倍不需要 Real-ESRGAN)"
        }
    }
}

# ── 5. 安装 sr_3dgs 本身 ──
echo ""
echo "安装 sr_3dgs..."
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# --no-build-isolation 跳过 setuptools 下载 (已有)
pip install -e "$PROJECT_DIR" --quiet --no-build-isolation 2>/dev/null || {
    echo "  --no-build-isolation 失败，尝试普通安装..."
    if [ -n "$MIRROR" ]; then
        pip install -e "$PROJECT_DIR" --quiet -i "$MIRROR" --trusted-host "$(echo $MIRROR | cut -d/ -f3)"
    else
        pip install -e "$PROJECT_DIR" --quiet
    fi
}

# ── 6. 验证 ──
echo ""
echo "========================================"
echo "  验证环境"
echo "========================================"

python -c "
import sys
print('  Python:', sys.version.split()[0])

import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()}', end='')
if torch.cuda.is_available():
    print(f' ({torch.cuda.get_device_name(0)})')
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    if mem > 0:
        mem_gb = mem / (1024**3)
        print(f'  VRAM:    {mem_gb:.1f} GB')
else:
    print('')

import numpy, cv2
print(f'  numpy:   {numpy.__version__}')
print(f'  opencv:  {cv2.__version__}')

try:
    import gsplat;   print(f'  gsplat:  OK')
except ImportError:
    print('  gsplat:  MISSING')

try:
    import realesrgan; print('  realesrgan: OK')
except ImportError:
    print('  realesrgan: N/A')

try:
    import sr_3dgs;  print(f'  sr_3dgs: OK')
except ImportError:
    print('  sr_3dgs: MISSING')

import shutil
for tool in ['colmap', 'ffmpeg']:
    path = shutil.which(tool)
    mark = 'OK' if path else 'MISSING'
    print(f'  {tool}:    {mark}')
" || true

echo ""
echo "========================================"
echo "  配置完成!"
echo ""
echo "  激活环境:  conda activate ${ENV_NAME}"
echo "  如果 conda activate 报错:"
echo "    ${CONDA_BASE}/bin/conda init bash"
echo "    source ~/.bashrc"
echo ""
echo "  运行管线:"
echo "    python scripts/run_video_pipeline.py --video input.mp4"
echo "========================================"
