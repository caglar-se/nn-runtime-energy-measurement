# nn-runtime-energy-measurement

Measures per-inference **runtime**, **energy**, and **average power** for ONNX models on GPU and CPU using hardware energy counters.

These scripts were developed for and used in the following JVET contribution:

> **Runtime and Energy Consumption Estimation for Neural Network-based In-loop Filters in NNVC**  
> https://jvet-experts.org/index.php?document=17130

---

## Requirements

| Script | Hardware | OS |
|--------|----------|----|
| `measure_gpu.py` | NVIDIA GPU (Maxwell or newer) | Linux |
| `measure_cpu.py` | Intel CPU with RAPL | Linux |
| `measure_cpu_macos.py` | Apple Silicon (arm64) | macOS |

```bash
# GPU
pip install -r requirements_gpu.txt

# CPU (Linux, RAPL)
pip install -r requirements_cpu.txt

# CPU (macOS, powermetrics) — same dependencies as above
pip install -r requirements_cpu.txt
```

---

## GPU — `measure_gpu.py`

### Energy measurement

- **Primary:** `nvmlDeviceGetTotalEnergyConsumption()` — hardware energy counter (mJ resolution), used automatically when supported.
- **Fallback:** background thread polls `nvmlDeviceGetPowerUsage()` at `--power_sample_dt` intervals and integrates with the trapezoidal rule.

Net energy per inference subtracts idle baseline power:

```
net_energy = gross_energy / n_runs  −  baseline_power × runtime / n_runs
```

### Usage

```bash
# Direct
python measure_gpu.py --models_dir /path/to/models --gpu_index 0

# With IOBinding (recommended for layer-by-layer sub-model benchmarks)
python measure_gpu.py --models_dir /path/to/models --gpu_index 0 --iobinding

# Via wrapper (auto-sets LD_LIBRARY_PATH for CUDA libraries)
bash run_measurement_gpu.sh --models_dir /path/to/models --gpu_index 0
```

> **Do not set `CUDA_VISIBLE_DEVICES`.** NVML uses physical device indices and ignores that variable. Use `--gpu_index <N>` to target the same physical device for both ORT and NVML.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--models_dir` | *(required)* | Directory containing `.onnx` files |
| `--output_csv` | `output/output_<dir>_<timestamp>.csv` | Output CSV path |
| `--gpu_index` | `0` | Physical GPU index (used by both ORT and NVML) |
| `--warmup_duration` | `3.0` | Warm-up duration in seconds |
| `--silence_duration` | `3.0` | Idle duration in seconds for baseline power measurement |
| `--measure_duration` | `3.0` | Measurement duration in seconds |
| `--power_sample_dt` | `0.01` | Sampling interval (s) for fallback power integration |
| `--iobinding` | off | Keep output tensors on GPU, eliminating PCIe transfer overhead |
| `--override_hw H W` | off | Override H and W for all 4-D model inputs |
| `--recursive` | off | Search sub-directories for `.onnx` files |
| `--verbose` | off | Print input shapes and debug information |

---

## CPU — `measure_cpu.py`

### Energy measurement

Reads the Linux RAPL hardware energy counter:

```
/sys/class/powercap/intel-rapl/intel-rapl:<package>/energy_uj
```

Counter wrap-around is handled automatically. Net energy uses the same baseline-subtraction formula as the GPU script.

### RAPL permissions

The energy counter file is root-readable by default. Grant access with:

```bash
sudo chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```

### Usage

```bash
# Direct
python measure_cpu.py --models_dir /path/to/models --rapl_package 0 --num_threads 1

# Pin to cores on the same NUMA node as the RAPL package (reduces cross-socket noise)
taskset -c 0-7 python measure_cpu.py --models_dir /path/to/models --rapl_package 0

# Via wrapper
bash run_measurement_cpu.sh --models_dir /path/to/models --rapl_package 0
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--models_dir` | *(required)* | Directory containing `.onnx` files |
| `--output_csv` | `output/output_cpu_<dir>_<timestamp>.csv` | Output CSV path |
| `--rapl_package` | `0` | RAPL package index (corresponds to CPU socket) |
| `--num_threads` | `10` | ORT intra/inter-op thread count |
| `--warmup_duration` | `3.0` | Warm-up duration in seconds |
| `--silence_duration` | `3.0` | Idle duration in seconds for baseline power measurement |
| `--measure_duration` | `3.0` | Measurement duration in seconds |
| `--override_hw H W` | off | Override H and W for all 4-D model inputs |
| `--recursive` | off | Search sub-directories for `.onnx` files |
| `--verbose` | off | Print input shapes and debug information |

---

## CPU (macOS) — `measure_cpu_macos.py`

### Energy measurement

Samples CPU package power via `sudo powermetrics --samplers cpu_power` in a background
thread and integrates average watts over the measurement window. Apple Silicon exposes no
RAPL-style monotonic energy counter, so this is a power-sampling approach rather than a
hardware energy counter — same net-energy baseline-subtraction formula as the other scripts.

### powermetrics permissions

`powermetrics` requires root. Grant passwordless access with:

```bash
sudo sh -c 'echo "'"$USER"' ALL=(ALL) NOPASSWD: /usr/bin/powermetrics" \
    > /etc/sudoers.d/powermetrics && chmod 440 /etc/sudoers.d/powermetrics'
```

### Usage

```bash
# Direct
python measure_cpu_macos.py --models_dir /path/to/models --num_threads 1

# Via wrapper
bash run_measurement_cpu_macos.sh --models_dir /path/to/models
```

> **macOS sleep:** long `powermetrics` sampling windows can be silently interrupted if the
> system sleeps mid-measurement. Wrap long-running invocations with `caffeinate -dimsu -w <pid>`
> (see `macbook/table2/run_measurements.sh` for an example).

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--models_dir` | *(required)* | Directory containing `.onnx` files |
| `--output_csv` | `output/output_cpu_macos_<dir>_<timestamp>.csv` | Output CSV path |
| `--pm_interval_ms` | `200` | `powermetrics` sampling interval in milliseconds |
| `--num_threads` | `10` | ORT intra/inter-op thread count |
| `--warmup_duration` | `3.0` | Warm-up duration in seconds |
| `--silence_duration` | `3.0` | Idle duration in seconds for baseline power measurement |
| `--measure_duration` | `3.0` | Measurement duration in seconds |
| `--override_hw H W` | off | Override H and W for all 4-D model inputs |
| `--recursive` | off | Search sub-directories for `.onnx` files |
| `--verbose` | off | Print input shapes and debug information |

Requires the same packages as `measure_cpu.py`: `pip install -r requirements_cpu.txt`.

> **`--override_hw` on models with fixed spatial dims:** some exported ONNX
> models hardcode H/W as fixed ints rather than dynamic axes, so ORT rejects
> a differently-shaped input at the API boundary. When `--override_hw` is
> given, `measure_cpu_macos.py` first rewrites the graph's declared input
> dims to symbolic (`make_spatial_dynamic()`, using the `onnx` package —
> `pip install onnx`) before creating the session, so the override works
> even on fixed-shape exports. ORT may log a benign `VerifyOutputSizes`
> warning in this case (the graph's *output*-shape annotation is stale after
> the rewrite; the computed values are still correct) — suppressed here via
> `log_severity_level`.

---

## Output

All three scripts produce identical CSV columns:

| Column | Unit | Description |
|--------|------|-------------|
| `date_time` | — | Measurement timestamp |
| `model_name` | — | `.onnx` filename |
| `runtime_perinf_sec` | s | Mean wall-clock time per inference |
| `net_energy_perinf_joule` | J | Mean energy per inference (baseline-subtracted) |
| `net_averagepower_perinf_watt` | W | `net_energy / runtime` |
| `energy_perinf_joule` | J | Mean gross energy per inference |
| `averagepower_perinf_watt` | W | `gross_energy / runtime` |
| `baseline_power_watt` | W | Idle power measured during silence window |
| `silence_avg_temp_c` | °C | GPU temperature during silence (GPU only; NaN for CPU/CPU-macOS) |
| `inference_avg_temp_c` | °C | GPU temperature during inference (GPU only; NaN for CPU/CPU-macOS) |

---

## Examples

**`nvidiagpu-amdcpu/table1/`** — reproduces Table 1 from the JVET contribution: NNVC decoder
measurements (LOP7 off / on) combined with isolated LOP7 ONNX inference on CPU and GPU.

```bash
# Edit the path variables at the top of the script, then run from the repo root:
bash nvidiagpu-amdcpu/table1/run_measurements.sh
```

The script runs four steps: decoder measurements (both LOP7 conditions), CPU ONNX
measurements, GPU ONNX measurements, and a final aggregation step that prints the
complete table including derived rows (overhead, 24× single patch, ratios).

**`macbook/table2/`** — reproduces Table 2 from the JVET contribution: the same NNVC
decoder measurements (LOP7 off / on) combined with isolated LOP7 ONNX inference, on
macOS / Apple Silicon. CPU only — no GPU step, since Apple Silicon has no discrete GPU.

```bash
# Edit the path variables at the top of the script, then run from the repo root:
bash macbook/table2/run_measurements.sh
```

The script runs three steps: decoder measurements (both LOP7 conditions, via
`--energy_backend powermetrics`), isolated LOP7 CPU ONNX measurements (via
`measure_cpu_macos.py`), and a final aggregation step that prints the complete table
including derived rows (overhead, 24× single patch, ratio).

---

## Notes

- **Core pinning:** use `taskset -c <cores>` to reduce OS scheduling noise. For CPU measurements, pin to cores on the same socket as the RAPL package being read. (No `taskset` equivalent is used on macOS.)
- **IOBinding:** use `--iobinding` when benchmarking layer-by-layer sub-models to eliminate PCIe transfer overhead from the measurement.
- **Dynamic shapes:** unresolved symbolic dimensions are filled with defaults (`batch=1`, `H=224`, `W=224`). Use `--override_hw` to match the actual inference input size.
- **Measure duration:** aim for at least 100 inference passes to reduce run-to-run jitter. Increase `--measure_duration` for large or slow models.
- **macOS sleep:** long `powermetrics` sampling runs can be silently corrupted if the system sleeps mid-measurement. Wrap long-running scripts with `caffeinate -dimsu -w <pid>` (already done in `macbook/table2/run_measurements.sh`).
