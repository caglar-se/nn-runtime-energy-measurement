#!/usr/bin/env bash
# run_measurements.sh
#
# Reproduces Table 1: NVIDIA A40 GPU + AMD CPU decoder and isolated LOP7
# measurements on kodim19 512×768, QP=42, one frame.
#
# Run this script from the root of the nn-runtime-energy-measurement repo:
#
#   bash nvidiagpu-amdcpu/table1/run_measurements.sh
#
# All four steps write CSV files to nvidiagpu-amdcpu/table1/results/.
# After all steps complete, compute_table.py reads those CSVs and prints
# the full table.
#
# Prerequisites
# -------------
# 1. VTM NNVC source, built (EncoderAppStatic, DecoderAppStatic):
#      git clone https://vcgit.hhi.fraunhofer.de/jvet-ahg-nnvc/VVCSoftware_VTM.git
#    This repo (cfg/CTC_JPEGAI/kodim19_512x768_8bit_420.cfg, cfg/encoder_intra_nnvc.cfg)
#    ships with the NNVC ad-hoc group's own config set already.
# 2. kodim19 input YUV — bundled in this repo at data/kodim19_512x768_8bit_420.yuv
#    (INPUT_YUV below already points there by default).
# 3. LOP7 ONNX model:
#      https://tumde-my.sharepoint.com/:u:/g/personal/serdar_caglar_tum_de/IQBNh_HDZQZ4TplNY-bEDpcaAT8Njez7VgPjbEZJY9Y7uVU?e=UGHxEe
#    Download and place lop7_full_model.onnx in its own directory, then point
#    LOP7_MODEL_DIR below at that directory.
# 4. RAPL energy counter readable:
#      sudo chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:<package>/energy_uj
# 5. GPU packages installed:  pip install -r requirements_gpu.txt
#    CPU packages installed:  pip install -r requirements_cpu.txt
#
# Hardware used for the original measurements:
#   GPU  : NVIDIA A40  (gpu_index=4 on that machine)
#   CPU  : AMD (rapl_package=1, cores 40-49 on socket 1)

set -euo pipefail

# ---------------------------------------------------------------------------
# *** EDIT THESE VARIABLES TO MATCH YOUR SYSTEM ***
# ---------------------------------------------------------------------------

VTM_DIR=/path/to/VVCSoftware_VTM
INPUT_YUV="$(cd "$(dirname "$0")/../.." && pwd)/data/kodim19_512x768_8bit_420.yuv"
BASE_CFG=cfg/encoder_intra_nnvc.cfg
SEQ_CFG=cfg/CTC_JPEGAI/kodim19_512x768_8bit_420.cfg
LOP7_MODEL_DIR=/path/to/dir_containing_lop7.onnx

RAPL_PACKAGE=1      # CPU socket index
GPU_INDEX=0         # Physical GPU index — do NOT set CUDA_VISIBLE_DEVICES
TASKSET_CORES=0-7   # CPU cores on the same NUMA node as RAPL_PACKAGE
N_RUNS=10

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

RESULTS_DIR="$(dirname "$0")/results"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Table 1 reproduction — nn-runtime-energy-measurement"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="

# ---------------------------------------------------------------------------
# Step 1: NNVC decoder measurements — LOP7 off and on
# ---------------------------------------------------------------------------
# run_codec_energy.py runs the VTM DecoderAppStatic in a warm-up / silence /
# measure loop and writes one row per (condition × stage × outer run) to the
# CSV. Both conditions (lop7_off, lop7_on) are measured in the same invocation.
#
# Rows used for Table 1: stage == "decoder", conditions lop7_off and lop7_on.
# Net energy per row = cpu_energy_per_run_j - cpu_baseline_w * runtime_per_run_sec.

echo ""
echo "Step 1 / 4 — Decoder measurements (LOP7 off + on, $N_RUNS outer runs each)"

taskset -c "$TASKSET_CORES" python nvidiagpu-amdcpu/table1/run_codec_energy.py \
    --vtm_dir       "$VTM_DIR" \
    --input_yuv     "$INPUT_YUV" \
    --base_cfg      "$BASE_CFG" \
    --seq_cfg       "$SEQ_CFG" \
    --frames 1 --qp 42 \
    --runs          "$N_RUNS" \
    --rapl_package  "$RAPL_PACKAGE" \
    --gpu_index     "$GPU_INDEX" \
    --conditions lop7_off lop7_on \
    --output_csv    "$RESULTS_DIR/decoder.csv"

echo "  → $RESULTS_DIR/decoder.csv"

# ---------------------------------------------------------------------------
# Step 2: Isolated LOP7 on CPU — N_RUNS independent measurement runs
# ---------------------------------------------------------------------------
# measure_cpu.py runs ONNX Runtime inference on all .onnx files in the given
# directory and writes one row per model. Running it N_RUNS times produces
# N_RUNS independent measurements; compute_table.py averages across them.
#
# --num_threads 1 matches the single-thread decoder context.

echo ""
echo "Step 2 / 4 — Isolated LOP7 on CPU ($N_RUNS runs)"

for i in $(seq 1 "$N_RUNS"); do
    echo "  Run $i / $N_RUNS"
    taskset -c "$TASKSET_CORES" python measure_cpu.py \
        --models_dir    "$LOP7_MODEL_DIR" \
        --rapl_package  "$RAPL_PACKAGE" \
        --num_threads   1 \
        --output_csv    "$RESULTS_DIR/lop7_cpu_run${i}.csv"
done

echo "  → $RESULTS_DIR/lop7_cpu_run1.csv … lop7_cpu_run${N_RUNS}.csv"

# ---------------------------------------------------------------------------
# Step 3: Isolated LOP7 on GPU — N_RUNS independent measurement runs
# ---------------------------------------------------------------------------
# run_measurement_gpu.sh sets LD_LIBRARY_PATH for the pip-installed CUDA libs.
# --iobinding keeps output tensors on the GPU and removes PCIe transfer
# overhead from the measurement (matches the inference-only use case).

echo ""
echo "Step 3 / 4 — Isolated LOP7 on GPU ($N_RUNS runs)"

for i in $(seq 1 "$N_RUNS"); do
    echo "  Run $i / $N_RUNS"
    bash run_measurement_gpu.sh \
        --models_dir    "$LOP7_MODEL_DIR" \
        --gpu_index     "$GPU_INDEX" \
        --iobinding \
        --output_csv    "$RESULTS_DIR/lop7_gpu_run${i}.csv"
done

echo "  → $RESULTS_DIR/lop7_gpu_run1.csv … lop7_gpu_run${N_RUNS}.csv"

# ---------------------------------------------------------------------------
# Step 4: Compute and print Table 1
# ---------------------------------------------------------------------------
# compute_table.py aggregates all CSVs, computes derived rows (24× single
# patch, overhead = on − off, ratios), and prints the full table.
#
# --patches_per_frame 24: one 512×768 kodim19 frame contains 24 LOP7 patches.

echo ""
echo "Step 4 / 4 — Computing Table 1"

python nvidiagpu-amdcpu/table1/compute_table.py \
    --decoder_csv       "$RESULTS_DIR/decoder.csv" \
    --cpu_csvs          "$RESULTS_DIR"/lop7_cpu_run*.csv \
    --gpu_csvs          "$RESULTS_DIR"/lop7_gpu_run*.csv \
    --patches_per_frame 24 \
    --output_md         "$RESULTS_DIR/table1.md"
