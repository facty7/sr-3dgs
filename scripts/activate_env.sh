#!/bin/bash
# Source this script to activate the sr_3dgs environment with correct CUDA setup
# Usage: source scripts/activate_env.sh

if [ -f "$HOME/yes/etc/profile.d/conda.sh" ]; then
    source "$HOME/yes/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

conda activate sr_3dgs 2>/dev/null || true

# Fix CUDA_HOME for WSL (nvcc at /usr/bin, toolkit at /usr/lib/cuda)
CUDADIR="/home/facty/.cuda_hack"
if [ ! -f "$CUDADIR/bin/nvcc" ] && [ -f /usr/bin/nvcc ]; then
    mkdir -p "$CUDADIR/bin"
    ln -sf /usr/bin/nvcc "$CUDADIR/bin/nvcc"
    ln -sf /usr/lib/cuda/include "$CUDADIR/include"
    ln -sf /usr/lib/cuda/lib64 "$CUDADIR/lib64"
fi
export CUDA_HOME="$CUDADIR"
echo "[env] CUDA_HOME=$CUDA_HOME"
echo "[env] sr_3dgs ready"