"""
measure_cpu_macos.py
=====================
macOS (Apple Silicon) counterpart of measure_cpu.py.

Measures per-inference runtime and CPU package energy for all ONNX models
inside a given directory, then saves results to a CSV file.

Measurement strategy
--------------------
Primary : `sudo powermetrics --samplers cpu_power` sampled by a background
          thread. Each "CPU Power: <mW>" line is parsed and averaged over
          the measurement window to give average watts, then multiplied by
          elapsed time for total joules — same concept as the RAPL energy_uj
          counter on Linux, but powermetrics only exposes power samples, not
          a monotonic energy counter, so there is no wrap-around to handle.

Output CSV columns are identical to measure_cpu.py / measure_gpu.py so all
three results can be loaded together for comparison.
Temperature fields are NaN (the `cpu_power` sampler does not expose
per-package temperature, same limitation as RAPL on the Linux side).

Requires passwordless sudo for `powermetrics`:
    sudo sh -c 'echo "<user> ALL=(ALL) NOPASSWD: /usr/bin/powermetrics" \\
        > /etc/sudoers.d/powermetrics && chmod 440 /etc/sudoers.d/powermetrics'

Required pip packages
---------------------
    pip install -r requirements_cpu.txt
"""

import argparse
import logging
import re
import signal
import subprocess
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
# ONNX type → numpy dtype mapping  (identical to measure_cpu.py / measure_gpu.py)
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
# 1. File discovery  (identical to measure_cpu.py / measure_gpu.py)
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

def make_spatial_dynamic(model_path: Path) -> bytes:
    """Return ONNX model bytes with H and W input dims replaced by symbolic
    names, so a model exported with fixed spatial dims can still be run at
    an overridden --override_hw size. Needed because ORT validates a runtime
    input's shape against the graph's *declared* input shape; if that
    declaration is a fixed int, any other size is rejected even though the
    (typically shape-agnostic, convolutional) graph body could compute it."""
    import onnx
    model = onnx.load(str(model_path))
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if len(dims) == 4:
            for pos in (2, 3):  # H, W
                dims[pos].ClearField("dim_value")
                dims[pos].dim_param = "spatial"
    return model.SerializeToString()


def create_session(
    model_path: Path, num_threads: int, override_hw: Optional[tuple] = None
) -> ort.InferenceSession:
    """Load an ONNX model and create a CPU ORT session."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = num_threads

    model_source = make_spatial_dynamic(model_path) if override_hw is not None else str(model_path)
    if override_hw is not None:
        # The graph's output-shape annotation is stale after the input-dim
        # rewrite (it still reflects the original export size), so ORT logs
        # a VerifyOutputSizes warning on every single inference. The actual
        # computed output is correct; only the annotation is wrong. Quiet it.
        opts.log_severity_level = 3  # Error and above only

    session = ort.InferenceSession(
        model_source,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    active = session.get_providers()
    logger.debug("Active providers for '%s': %s", model_path.name, active)
    return session

# ---------------------------------------------------------------------------
# 3. Dummy input generation  (identical to measure_cpu.py / measure_gpu.py)
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
# 4. powermetrics energy sampling (macOS Apple Silicon CPU energy)
# ---------------------------------------------------------------------------

class PowermetricsReader:
    """Sample CPU package power via `sudo powermetrics` in a background thread."""

    def __init__(self, interval_ms: int = 200):
        self._interval_ms = interval_ms
        self._samples_mw: list = []
        self._proc = None
        self._thread = None

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "cpu_power",
             "-i", str(self._interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def _collect(self) -> None:
        for raw in self._proc.stdout:
            line = raw.decode("utf-8", errors="ignore")
            m = re.search(r"CPU Power:\s+(\d+)\s+mW", line)
            if m:
                self._samples_mw.append(float(m.group(1)))

    def stop_and_get_energy_j(self, elapsed_sec: float) -> float:
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self._samples_mw:
            logger.warning("powermetrics: no CPU Power samples collected.")
            return float("nan")
        avg_w = sum(self._samples_mw) / len(self._samples_mw) / 1000.0
        return avg_w * elapsed_sec


def check_powermetrics() -> None:
    try:
        r = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"],
            capture_output=True, timeout=5,
        )
        if r.returncode == 0:
            return
    except Exception:
        pass
    import getpass
    logger.error(
        "powermetrics is not runnable without a password prompt. Fix: "
        "sudo sh -c 'echo \"%s ALL=(ALL) NOPASSWD: /usr/bin/powermetrics\" "
        "> /etc/sudoers.d/powermetrics && chmod 440 /etc/sudoers.d/powermetrics'",
        getpass.getuser(),
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. Measurement functions
# ---------------------------------------------------------------------------

def measure_baseline_power(silence_duration: float, pm_interval_ms: int) -> float:
    """Measure idle CPU package power during a silence window. Returns watts."""
    pm = PowermetricsReader(pm_interval_ms)
    pm.start()
    time.sleep(silence_duration)
    energy_j = pm.stop_and_get_energy_j(silence_duration)
    return energy_j / silence_duration


def measure_with_energy_counter(
    session:          ort.InferenceSession,
    inputs:           Dict[str, np.ndarray],
    measure_duration: float,
    pm_interval_ms:   int,
) -> Tuple[float, float, int]:
    """Run inference for measure_duration seconds and return
    (total_time_sec, total_energy_joules, num_runs)."""
    pm = PowermetricsReader(pm_interval_ms)
    pm.start()

    n  = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < measure_duration:
        session.run(None, inputs)
        n += 1
    t1 = time.perf_counter()

    total_energy_j = pm.stop_and_get_energy_j(t1 - t0)
    return t1 - t0, total_energy_j, n

# ---------------------------------------------------------------------------
# 6. Per-model processing
# ---------------------------------------------------------------------------

def process_model(
    model_path: Path,
    args:       argparse.Namespace,
) -> Optional[Dict]:
    model_name = model_path.name
    logger.info("[%s] Loading session …", model_name)

    override_hw = tuple(args.override_hw) if getattr(args, "override_hw", None) else None

    try:
        session = create_session(model_path, args.num_threads, override_hw=override_hw)
    except Exception as exc:
        logger.warning("[%s] Session creation failed: %s", model_name, exc)
        return None

    try:
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
    baseline_power_w = measure_baseline_power(args.silence_duration, args.pm_interval_ms)
    logger.info("[%s] Baseline power: %.3f W", model_name, baseline_power_w)

    # --- Measured inference ---
    logger.info("[%s] Measuring (%.1f s) …", model_name, args.measure_duration)
    try:
        total_time, total_energy, n = measure_with_energy_counter(
            session, inputs, args.measure_duration, args.pm_interval_ms
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
        "silence_avg_temp_c":           float("nan"),   # powermetrics cpu_power has no temp
        "inference_avg_temp_c":         float("nan"),
    }

# ---------------------------------------------------------------------------
# 7. CSV output  (identical to measure_cpu.py / measure_gpu.py)
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
            "Benchmark ONNX models on CPU (macOS / Apple Silicon) using powermetrics: "
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
        help="Destination CSV file. Defaults to output/output_cpu_macos_<models_dir_name>_<timestamp>.csv",
    )
    parser.add_argument(
        "--pm_interval_ms", type=int, default=200,
        help="powermetrics sampling interval in milliseconds.",
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
        args.output_csv = Path("output") / f"output_cpu_macos_{args.models_dir.name}_{timestamp}.csv"
        logger.info("Output file: %s", args.output_csv)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- powermetrics pre-flight check ---
    check_powermetrics()

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
        result = process_model(model_path, args)
        if result is not None:
            append_result(result, args.output_csv)

    logger.info("Done. Results saved to '%s'.", args.output_csv)


if __name__ == "__main__":
    main()
