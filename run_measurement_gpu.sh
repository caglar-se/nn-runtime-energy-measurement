#!/usr/bin/env bash
# run_energy.sh — wrapper for measure_gpu.py
#
# Sets LD_LIBRARY_PATH so that onnxruntime-gpu can locate the CUDA 12
# libraries shipped with the nvidia-*-cu12 pip packages (libcublasLt.so.12,
# libcudnn.so.9, etc.).  All arguments are forwarded to the Python script.
#
# Usage examples:
#   bash run_energy.sh --models_dir /path/to/models --gpu_index 0
#   bash run_energy.sh --models_dir /path/to/models --gpu_index 0 --iobinding
#
# NOTE: Do NOT set CUDA_VISIBLE_DEVICES.
# NVML uses physical device indices and ignores CUDA_VISIBLE_DEVICES.
# Pass --gpu_index <N> so both ORT and NVML target the same physical GPU.
#
# Optional: pin the process to specific CPU cores with taskset to reduce
# OS scheduling noise, e.g.:
#   taskset -c 0-7 bash run_energy.sh --models_dir /path/to/models --gpu_index 0

set -euo pipefail

# Locate the site-packages directory that holds the nvidia-* packages.
NVIDIA_SP=$(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia

export LD_LIBRARY_PATH="\
${NVIDIA_SP}/cublas/lib:\
${NVIDIA_SP}/cuda_runtime/lib:\
${NVIDIA_SP}/cuda_nvrtc/lib:\
${NVIDIA_SP}/cudnn/lib:\
${NVIDIA_SP}/cufft/lib:\
${NVIDIA_SP}/curand/lib:\
${NVIDIA_SP}/cusolver/lib:\
${NVIDIA_SP}/cusparse/lib:\
${NVIDIA_SP}/nvjitlink/lib:\
${NVIDIA_SP}/nccl/lib:\
${LD_LIBRARY_PATH:-}"

echo "Running: python measure_gpu.py $*"
exec python measure_gpu.py "$@"
