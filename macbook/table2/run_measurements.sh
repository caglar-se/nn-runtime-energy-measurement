#!/usr/bin/env bash
# run_measurements.sh
#
# Reproduces Table 2: MacBook Apple Silicon CPU decoder and isolated LOP7
# measurements on kodim19 512×768, QP=42, one frame. No GPU section —
# Apple Silicon has no discrete GPU.
#
# Run this script from the root of the nn-runtime-energy-measurement repo:
#
#   bash macbook/table2/run_measurements.sh
#
# All three steps write CSV files to macbook/table2/results/.
# After all steps complete, compute_table.py reads those CSVs and prints
# the full table.
#
# Prerequisites
# -------------
# 1. VTM NNVC binary built for macOS/arm64 (EncoderAppStatic, DecoderAppStatic) —
#      cmake -DCMAKE_BUILD_TYPE=Release -B build_mac
#      cmake --build build_mac -j$(sysctl -n hw.logicalcpu)
#    then copy the built binaries into <VTM_DIR>/bin/.
# 2. kodim19 input YUV and matching sequence config (.cfg) available.
# 3. LOP7 ONNX model file in a dedicated directory.
# 4. Passwordless sudo for powermetrics:
#      sudo sh -c 'echo "'"$USER"' ALL=(ALL) NOPASSWD: /usr/bin/powermetrics" \
#        > /etc/sudoers.d/powermetrics && chmod 440 /etc/sudoers.d/powermetrics'
# 5. CPU packages installed:  pip install -r requirements_cpu.txt
# 6. The `onnx` package (pip install onnx) — needed by measure_cpu_macos.py's
#    --override_hw to rewrite fixed-shape model inputs to dynamic (see the
#    README's "CPU (macOS)" section for why this is required).
#
# Hardware used for the original measurements:
#   CPU  : Apple Silicon (MacBook), powermetrics cpu_power sampler
#
# Note on macOS sleep: powermetrics sampling runs can be silently corrupted
# if the system sleeps mid-measurement. This script wraps itself in
# `caffeinate` below to prevent that.

set -euo pipefail

# Keep the system awake for the lifetime of this script (auto-exits with it).
caffeinate -dimsu -w $$ &

# ---------------------------------------------------------------------------
# *** EDIT THESE VARIABLES TO MATCH YOUR SYSTEM ***
# ---------------------------------------------------------------------------

VTM_DIR=/Users/serdarcaglar/workspace/nn_energy/jvet-jul26/VVCSoftware_VTM
INPUT_YUV=/Users/serdarcaglar/workspace/nn_energy/jvet-jul26/VVCSoftware_VTM/CTC_Image/yuv_output_ctc/kodim19_512x768_8bit_420.yuv
BASE_CFG=cfg/encoder_intra_nnvc.cfg
SEQ_CFG=cfg/CTC_JPEGAI/kodim19_512x768_8bit_420.cfg
LOP7_MODEL_DIR=/Users/serdarcaglar/workspace/nn_energy/jvet-jul26/models/NN_Filtering_Models_onnx/LOP7

N_RUNS=10

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

RESULTS_DIR="$(dirname "$0")/results"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Table 2 reproduction — nn-runtime-energy-measurement (macOS)"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="

# ---------------------------------------------------------------------------
# Step 1: NNVC decoder measurements — LOP7 off and on
# ---------------------------------------------------------------------------
# run_codec_energy.py runs the VTM DecoderAppStatic in a warm-up / silence /
# measure loop and writes one row per (condition × stage × outer run) to the
# CSV. Both conditions (lop7_off, lop7_on) are measured in the same invocation.
# --gpu_index -1 disables NVML (no discrete GPU on Apple Silicon).
# --energy_backend powermetrics uses the macOS CPU power sampler.
#
# Rows used for Table 2: stage == "decoder", conditions lop7_off and lop7_on.
# Net energy per row = cpu_energy_per_run_j - cpu_baseline_w * runtime_per_run_sec.

echo ""
echo "Step 1 / 3 — Decoder measurements (LOP7 off + on, $N_RUNS outer runs each)"

python macbook/table2/run_codec_energy.py \
    --vtm_dir       "$VTM_DIR" \
    --input_yuv     "$INPUT_YUV" \
    --base_cfg      "$BASE_CFG" \
    --seq_cfg       "$SEQ_CFG" \
    --frames 1 --qp 42 \
    --runs          "$N_RUNS" \
    --gpu_index     -1 \
    --energy_backend powermetrics \
    --conditions lop7_off lop7_on \
    --output_csv    "$RESULTS_DIR/decoder.csv"

echo "  → $RESULTS_DIR/decoder.csv"

# ---------------------------------------------------------------------------
# Step 2: Isolated LOP7 on CPU — N_RUNS independent measurement runs
# ---------------------------------------------------------------------------
# measure_cpu_macos.py runs ONNX Runtime inference on all .onnx files in the
# given directory and writes one row per model. Running it N_RUNS times
# produces N_RUNS independent measurements; compute_table.py averages across
# them.
#
# --num_threads 1 --override_hw 72 72 matches the "fair setup": the decoder
# calls LOP7 single-threaded on 72×72 patches (post-DCT-transform block size),
# not the model's native 144×144 export resolution. lop7_full_model.onnx has
# H/W hardcoded (not dynamic axes), so measure_cpu_macos.py rewrites the
# graph's declared input dims to symbolic before creating the session
# whenever --override_hw is given (see make_spatial_dynamic()) — this is the
# same technique used to produce the original macOS reference numbers.

echo ""
echo "Step 2 / 3 — Isolated LOP7 on CPU ($N_RUNS runs)"

for i in $(seq 1 "$N_RUNS"); do
    echo "  Run $i / $N_RUNS"
    python measure_cpu_macos.py \
        --models_dir    "$LOP7_MODEL_DIR" \
        --num_threads   1 \
        --override_hw   72 72 \
        --output_csv    "$RESULTS_DIR/lop7_cpu_run${i}.csv"
done

echo "  → $RESULTS_DIR/lop7_cpu_run1.csv … lop7_cpu_run${N_RUNS}.csv"

# ---------------------------------------------------------------------------
# Step 3: Compute and print Table 2
# ---------------------------------------------------------------------------
# compute_table.py aggregates all CSVs, computes derived rows (24× single
# patch, overhead = on − off, ratio), and prints the full table.
#
# --patches_per_frame 24: one 512×768 kodim19 frame contains 24 LOP7 patches.

echo ""
echo "Step 3 / 3 — Computing Table 2"

python macbook/table2/compute_table.py \
    --decoder_csv       "$RESULTS_DIR/decoder.csv" \
    --cpu_csvs          "$RESULTS_DIR"/lop7_cpu_run*.csv \
    --patches_per_frame 24 \
    --output_md         "$RESULTS_DIR/table2.md"
