"""
analyse_cpu_energy.py
=====================
CPU counterpart of analyse_nn_energy.py.

Measures per-inference runtime and CPU package energy for all ONNX models
inside a given directory, then saves results to a CSV file.

Measurement strategy
--------------------
Primary : Linux RAPL (Running Average Power Limit) via sysfs powercap interface
          /sys/class/powercap/intel-rapl/intel-rapl:<package>/energy_uj
          Reads a hardware energy counter in microjoules — same concept as
          nvmlDeviceGetTotalEnergyConsumption() on the GPU side.

The counter wraps at max_energy_range_uj; wrap-around is handled explicitly.

Output CSV columns are identical to analyse_nn_energy.py so both results can
be loaded together for GPU vs CPU comparison.
Temperature fields are NaN (RAPL does not expose per-package temperature;
add hwmon reads if needed).

Required pip packages
---------------------
    pip install onnxruntime numpy pandas
"""

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ONNX type → numpy dtype mapping  (identical to GPU script)
# ---------------------------------------------------------------------------

ONNX_DTYPE_MAP: Dict[str, type] = {
    "tensor(float)":   np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)":  np.float64,
    "tensor(int64)":   np.int64,
    "tensor(int32)":   np.int32,
    "tensor(int16)":   np.int16,
    "tensor(int8)":    np.int8,
    "tensor(uint64)":  np.uint64,
    "tensor(uint32)":  np.uint32,
    "tensor(uint16)":  np.uint16,
    "tensor(uint8)":   np.uint8,
    "tensor(bool)":    np.bool_,
    "tensor(string)":  None,
}

DEFAULT_BATCH_SIZE = 1
DEFAULT_CHANNEL    = 3
DEFAULT_HEIGHT     = 224
DEFAULT_WIDTH      = 224
DEFAULT_SEQ_LEN    = 128

# ---------------------------------------------------------------------------
# 1. File discovery  (identical to GPU script)
# ---------------------------------------------------------------------------

def find_onnx_files(models_dir: Path, recursive: bool = False) -> List[Path]:
    pattern = "**/*.onnx" if recursive else "*.onnx"
    files = sorted(models_dir.glob(pattern))
    if not files:
        logger.warning(
            "No .onnx files found in '%s' (recursive=%s).", models_dir, recursive
        )
    return files

# ---------------------------------------------------------------------------
# 2. CPU ONNX Runtime session
# ---------------------------------------------------------------------------

def create_session(model_path: Path, num_threads: int) -> ort.InferenceSession:
    """Load an ONNX model and create a CPU ORT session."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = num_threads

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    active = session.get_providers()
    logger.debug("Active providers for '%s': %s", model_path.name, active)
    return session

# ---------------------------------------------------------------------------
# 3. Dummy input generation  (identical to GPU script)
# ---------------------------------------------------------------------------

def _resolve_dim(dim, position: int, total_dims: int, batch_size: int) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    if position == 0:
        return batch_size
    if total_dims == 4:
        return {1: DEFAULT_CHANNEL, 2: DEFAULT_HEIGHT, 3: DEFAULT_WIDTH}.get(position, 1)
    if total_dims == 3 and position == 1:
        return DEFAULT_SEQ_LEN
    return 1


def make_dummy_inputs(
    session: ort.InferenceSession,
    batch_size: int = DEFAULT_BATCH_SIZE,
    override_hw: Optional[tuple] = None,
) -> Dict[str, np.ndarray]:
    dummy: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        name      = inp.name
        shape     = inp.shape
        dtype_str = inp.type
        np_dtype  = ONNX_DTYPE_MAP.get(dtype_str)
        if np_dtype is None:
            raise ValueError(
                f"Input '{name}' has unsupported type '{dtype_str}'."
            )
        n_dims   = len(shape)
        resolved = [_resolve_dim(d, i, n_dims, batch_size) for i, d in enumerate(shape)]
        if override_hw is not None and n_dims == 4:
            resolved[2] = override_hw[0]
            resolved[3] = override_hw[1]
        dummy[name] = np.zeros(resolved, dtype=np_dtype)
        logger.debug("  input '%s': shape=%s  dtype=%s", name, resolved, np_dtype)
    return dummy

# ---------------------------------------------------------------------------
# 4. RAPL energy counter
# ---------------------------------------------------------------------------

RAPL_BASE = Path("/sys/class/powercap/intel-rapl")


def _rapl_path(package: int) -> Path:
    return RAPL_BASE / f"intel-rapl:{package}"


def read_rapl_uj(package: int) -> int:
    """Read current RAPL package energy in microjoules."""
    return int((_rapl_path(package) / "energy_uj").read_text().strip())


def rapl_max_uj(package: int) -> int:
    """Read the counter wrap-around limit in microjoules."""
    return int((_rapl_path(package) / "max_energy_range_uj").read_text().strip())


def rapl_delta_uj(start: int, end: int, max_uj: int) -> int:
    """Compute energy delta handling counter wrap-around."""
    if end >= start:
        return end - start
    return max_uj - start + end  # wrap


def check_rapl(package: int) -> None:
    path = _rapl_path(package)
    if not path.exists():
        logger.error("RAPL package %d not found at %s", package, path)
        sys.exit(1)
    try:
        read_rapl_uj(package)
    except PermissionError:
        logger.error(
            "Cannot read %s/energy_uj — run as root or: "
            "chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:%d/energy_uj",
            path, package,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# 5. Measurement functions
# ---------------------------------------------------------------------------

def measure_baseline_power(
    package:          int,
    silence_duration: float,
    max_uj:           int,
) -> float:
    """Measure idle CPU package power during a silence window. Returns watts."""
    start_uj = read_rapl_uj(package)
    time.sleep(silence_duration)
    end_uj = read_rapl_uj(package)
    delta_j = rapl_delta_uj(start_uj, end_uj, max_uj) * 1e-6
    return delta_j / silence_duration


def measure_with_energy_counter(
    session:          ort.InferenceSession,
    inputs:           Dict[str, np.ndarray],
    measure_duration: float,
    package:          int,
    max_uj:           int,
) -> Tuple[float, float, int]:
    """Run inference for measure_duration seconds and return
    (total_time_sec, total_energy_joules, num_runs)."""
    start_uj = read_rapl_uj(package)

    n  = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < measure_duration:
        session.run(None, inputs)
        n += 1
    t1 = time.perf_counter()

    end_uj = read_rapl_uj(package)
    total_energy_j = rapl_delta_uj(start_uj, end_uj, max_uj) * 1e-6
    return t1 - t0, total_energy_j, n

# ---------------------------------------------------------------------------
# 6. Per-model processing
# ---------------------------------------------------------------------------

def process_model(
    model_path: Path,
    args:       argparse.Namespace,
    max_uj:     int,
) -> Optional[Dict]:
    model_name = model_path.name
    logger.info("[%s] Loading session …", model_name)

    try:
        session = create_session(model_path, args.num_threads)
    except Exception as exc:
        logger.warning("[%s] Session creation failed: %s", model_name, exc)
        return None

    try:
        override_hw = tuple(args.override_hw) if getattr(args, "override_hw", None) else None
        inputs = make_dummy_inputs(session, batch_size=DEFAULT_BATCH_SIZE, override_hw=override_hw)
    except Exception as exc:
        logger.warning("[%s] Could not build dummy inputs: %s", model_name, exc)
        return None

    if args.verbose:
        for inp_name, arr in inputs.items():
            logger.info("  input '%s': shape=%s  dtype=%s", inp_name, arr.shape, arr.dtype)

    # --- Warm-up ---
    logger.info("[%s] Warm-up (%.1f s) …", model_name, args.warmup_duration)
    try:
        t_warmup = time.perf_counter()
        while time.perf_counter() - t_warmup < args.warmup_duration:
            session.run(None, inputs)
    except Exception as exc:
        logger.warning("[%s] Warm-up failed: %s", model_name, exc)
        return None

    # --- Silence (baseline power) ---
    logger.info("[%s] Silence (%.1f s) …", model_name, args.silence_duration)
    baseline_power_w = measure_baseline_power(
        args.rapl_package, args.silence_duration, max_uj
    )
    logger.info("[%s] Baseline power: %.3f W", model_name, baseline_power_w)

    # --- Measured inference ---
    logger.info("[%s] Measuring (%.1f s) …", model_name, args.measure_duration)
    try:
        total_time, total_energy, n = measure_with_energy_counter(
            session, inputs, args.measure_duration, args.rapl_package, max_uj
        )
    except Exception as exc:
        logger.warning("[%s] Measurement failed: %s", model_name, exc)
        return None

    logger.info("[%s] Completed %d runs in %.2f s.", model_name, n, total_time)

    runtime_perinf      = total_time / n
    energy_perinf       = total_energy / n
    avgpower_perinf     = energy_perinf / runtime_perinf if runtime_perinf > 0 else float("nan")
    net_energy_perinf   = energy_perinf - baseline_power_w * runtime_perinf
    net_avgpower_perinf = net_energy_perinf / runtime_perinf if runtime_perinf > 0 else float("nan")

    logger.info(
        "[%s]  runtime/inf=%.6f s  |  energy/inf=%.6f J  |  avgpower/inf=%.3f W  |  "
        "net_energy/inf=%.6f J  |  net_avgpower/inf=%.3f W  |  baseline=%.3f W",
        model_name, runtime_perinf, energy_perinf, avgpower_perinf,
        net_energy_perinf, net_avgpower_perinf, baseline_power_w,
    )

    return {
        "date_time":                    time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name":                   model_name,
        "runtime_perinf_sec":           runtime_perinf,
        "net_energy_perinf_joule":      net_energy_perinf,
        "net_averagepower_perinf_watt": net_avgpower_perinf,
        "energy_perinf_joule":          energy_perinf,
        "averagepower_perinf_watt":     avgpower_perinf,
        "baseline_power_watt":          baseline_power_w,
        "silence_avg_temp_c":           float("nan"),   # RAPL has no temp
        "inference_avg_temp_c":         float("nan"),
    }

# ---------------------------------------------------------------------------
# 7. CSV output  (identical to GPU script)
# ---------------------------------------------------------------------------

COLUMNS = [
    "date_time",
    "model_name",
    "runtime_perinf_sec",
    "net_energy_perinf_joule",
    "net_averagepower_perinf_watt",
    "energy_perinf_joule",
    "averagepower_perinf_watt",
    "baseline_power_watt",
    "silence_avg_temp_c",
    "inference_avg_temp_c",
]


def append_result(result: Dict, output_csv: Path) -> None:
    write_header = not output_csv.exists()
    df = pd.DataFrame([result], columns=COLUMNS)
    df.to_csv(output_csv, mode="a", index=False, header=write_header)
    logger.info("Appended result for '%s' to '%s'.", result["model_name"], output_csv)

# ---------------------------------------------------------------------------
# 8. CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ONNX models on CPU using RAPL energy counters: "
            "measures per-inference runtime, energy, and average power."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models_dir", required=True, type=Path,
        help="Directory containing .onnx model files.",
    )
    parser.add_argument(
        "--output_csv", type=Path, default=None,
        help="Destination CSV file. Defaults to output/output_cpu_<models_dir_name>_<timestamp>.csv",
    )
    parser.add_argument(
        "--rapl_package", type=int, default=0,
        help="RAPL package index to read energy from (matches the CPU socket pinned by taskset).",
    )
    parser.add_argument(
        "--num_threads", type=int, default=10,
        help="ORT intra/inter op thread count.",
    )
    parser.add_argument(
        "--warmup_duration", type=float, default=3.0,
        help="Duration (seconds) of warm-up inference before silence phase.",
    )
    parser.add_argument(
        "--silence_duration", type=float, default=3.0,
        help="Duration (seconds) of CPU silence to measure baseline power.",
    )
    parser.add_argument(
        "--measure_duration", type=float, default=3.0,
        help="Duration (seconds) to run measured inference passes.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Scan sub-directories recursively for .onnx files.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print additional debug information.",
    )
    parser.add_argument(
        "--override_hw", type=int, nargs=2, default=None, metavar=("H", "W"),
        help="Override spatial H and W dimensions for all 4-D model inputs (e.g. --override_hw 72 72).",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# 9. Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.models_dir.is_dir():
        logger.error("--models_dir '%s' is not a directory.", args.models_dir)
        sys.exit(1)

    if args.output_csv is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_csv = Path("output") / f"output_cpu_{args.models_dir.name}_{timestamp}.csv"
        logger.info("Output file: %s", args.output_csv)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- RAPL pre-flight check ---
    check_rapl(args.rapl_package)
    max_uj = rapl_max_uj(args.rapl_package)
    rapl_name = (_rapl_path(args.rapl_package) / "name").read_text().strip()
    logger.info(
        "RAPL package %d: %s  (max_energy_range=%.1f kJ)",
        args.rapl_package, rapl_name, max_uj * 1e-9
    )

    onnx_files = find_onnx_files(args.models_dir, args.recursive)
    logger.info("Found %d ONNX file(s) to benchmark.", len(onnx_files))

    interrupted = threading.Event()

    def _sigint_handler(sig, frame):
        logger.warning("\nInterrupted by user — saving partial results …")
        interrupted.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    for i, model_path in enumerate(onnx_files, 1):
        if interrupted.is_set():
            break
        logger.info("--- [%d/%d] %s ---", i, len(onnx_files), model_path.name)
        result = process_model(model_path, args, max_uj)
        if result is not None:
            append_result(result, args.output_csv)

    logger.info("Done. Results saved to '%s'.", args.output_csv)


if __name__ == "__main__":
    main()
