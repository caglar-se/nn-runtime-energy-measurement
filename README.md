# nn-runtime-energy-measurement

Measures per-inference **runtime**, **energy**, and **average power** for ONNX models on GPU and CPU using hardware energy counters.

Developed for: **Runtime and Energy Consumption Estimation for Neural Network-based In-loop Filters in NNVC** — https://jvet-experts.org/index.php?document=17130

---

## Scripts

| Script | Hardware | OS | Energy source |
|---|---|---|---|
| `measure_gpu.py` | NVIDIA GPU (Maxwell+) | Linux | NVML energy counter |
| `measure_cpu.py` | Intel CPU with RAPL | Linux | RAPL energy counter |
| `measure_cpu_macos.py` | Apple Silicon | macOS | `powermetrics` power sampling |

```bash
pip install -r requirements_gpu.txt   # measure_gpu.py
pip install -r requirements_cpu.txt   # measure_cpu.py / measure_cpu_macos.py
```

Net energy per inference = gross energy − (baseline idle power × runtime) — same formula for all three scripts.

---

## Permissions

**RAPL** (Linux): `sudo chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj`

**powermetrics** (macOS), passwordless sudo:
```bash
sudo sh -c 'echo "'"$USER"' ALL=(ALL) NOPASSWD: /usr/bin/powermetrics" \
    > /etc/sudoers.d/powermetrics && chmod 440 /etc/sudoers.d/powermetrics'
```

---

## Usage

```bash
# GPU
python measure_gpu.py --models_dir /path/to/models --gpu_index 0 [--iobinding]
bash run_measurement_gpu.sh --models_dir /path/to/models --gpu_index 0   # sets CUDA LD_LIBRARY_PATH

# CPU (Linux)
python measure_cpu.py --models_dir /path/to/models --rapl_package 0 --num_threads 1
bash run_measurement_cpu.sh --models_dir /path/to/models --rapl_package 0

# CPU (macOS)
python measure_cpu_macos.py --models_dir /path/to/models --num_threads 1
bash run_measurement_cpu_macos.sh --models_dir /path/to/models
```

> Don't set `CUDA_VISIBLE_DEVICES` — NVML uses physical device indices and ignores it. Use `--gpu_index <N>` for both ORT and NVML.

---

## Arguments

Common to all three scripts:

| Argument | Default | Description |
|---|---|---|
| `--models_dir` | *(required)* | Directory containing `.onnx` files |
| `--output_csv` | auto-named with timestamp | Output CSV path |
| `--warmup_duration` / `--silence_duration` / `--measure_duration` | `3.0` each | Warm-up / idle-baseline / measurement window (seconds) |
| `--override_hw H W` | off | Override H/W for all 4-D model inputs |
| `--recursive` | off | Search sub-directories for `.onnx` files |
| `--verbose` | off | Print input shapes and debug info |

Script-specific:

| Argument | Script | Default | Description |
|---|---|---|---|
| `--gpu_index` | GPU | `0` | Physical GPU index (used by both ORT and NVML) |
| `--power_sample_dt` | GPU | `0.01` | Fallback power-integration sample interval (s) |
| `--iobinding` | GPU | off | Keep tensors on GPU (removes PCIe transfer overhead) |
| `--rapl_package` | CPU | `0` | RAPL package index (CPU socket) |
| `--num_threads` | CPU / CPU-macOS | `10` | ORT intra/inter-op thread count |
| `--pm_interval_ms` | CPU-macOS | `200` | `powermetrics` sampling interval (ms) |

**`--override_hw` on macOS:** if a model's ONNX export hardcodes H/W instead of using dynamic axes, `measure_cpu_macos.py` rewrites the graph's input dims to symbolic before loading (requires `pip install onnx`) so the override still works. ORT may log a harmless `VerifyOutputSizes` warning when it does this.

---

## Output

All three scripts write the same CSV columns:

| Column | Unit | Description |
|---|---|---|
| `date_time` | — | Timestamp |
| `model_name` | — | `.onnx` filename |
| `runtime_perinf_sec` | s | Mean time per inference |
| `net_energy_perinf_joule` | J | Mean energy per inference (baseline-subtracted) |
| `net_averagepower_perinf_watt` | W | `net_energy / runtime` |
| `energy_perinf_joule` | J | Mean gross energy per inference |
| `averagepower_perinf_watt` | W | `gross_energy / runtime` |
| `baseline_power_watt` | W | Idle power during silence window |
| `silence_avg_temp_c` / `inference_avg_temp_c` | °C | GPU temperature (GPU only; NaN for CPU/CPU-macOS) |

---

## Examples

- **`nvidiagpu-amdcpu/table1/`** — reproduces Table 1 (NVIDIA GPU + AMD CPU): decoder LOP7 off/on, plus isolated LOP7 on CPU and GPU.
  ```bash
  bash nvidiagpu-amdcpu/table1/run_measurements.sh   # edit paths at the top first
  ```
- **`macbook/table2/`** — reproduces Table 2 (Apple Silicon, CPU only — no discrete GPU): same decoder measurement, plus isolated LOP7 on CPU via `measure_cpu_macos.py`.
  ```bash
  bash macbook/table2/run_measurements.sh   # edit paths at the top first
  ```

Each script runs its measurement steps in order, then prints and saves the aggregated table (overhead, N× single-patch, ratios).

---

## Notes

- **Core pinning (Linux only):** `taskset -c <cores>` reduces scheduling noise; pin to the socket matching `--rapl_package`.
- **Dynamic shapes:** unresolved dims default to `batch=1, H=224, W=224` — use `--override_hw` to match your real input size.
- **Measure duration:** aim for 100+ inference passes; increase `--measure_duration` for slow models.
- **macOS sleep:** wrap long `powermetrics` runs in `caffeinate -dimsu -w <pid>` to avoid silent corruption (already done in `macbook/table2/run_measurements.sh`).
