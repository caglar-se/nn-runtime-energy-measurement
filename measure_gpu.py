"""
analyse_nn_energy.py
====================
Measures inference runtime, GPU energy, and average GPU power for all ONNX
models inside a given directory, then saves the results to a CSV file.

Measurement strategy
--------------------
Primary  : nvmlDeviceGetTotalEnergyConsumption()  (mJ → J)
Fallback : background thread samples nvmlDeviceGetPowerUsage() (mW) at a
           configurable interval and integrates with the trapezoidal rule.

ONNX Runtime / CUDA synchronisation note
-----------------------------------------
OrtSession.run() with CUDAExecutionProvider blocks until the GPU computation
is finished and the output tensors have been copied back to host memory.
Therefore no explicit cudaDeviceSynchronize call is required; timing brackets
around session.run() already capture the true GPU-inclusive latency.

Required pip packages
---------------------
    pip install onnxruntime-gpu nvidia-ml-py numpy pandas
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
import pynvml
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
# ONNX type → numpy dtype mapping
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
    "tensor(string)":  None,   # strings: cannot generate numeric dummy
}

# Default spatial/sequence dimensions when a dim cannot be inferred.
DEFAULT_BATCH_SIZE = 1
DEFAULT_CHANNEL    = 3
DEFAULT_HEIGHT     = 224
DEFAULT_WIDTH      = 224
DEFAULT_SEQ_LEN    = 128   # for NLP-style sequence dimensions


# ---------------------------------------------------------------------------
# 1. File discovery
# ---------------------------------------------------------------------------

def find_onnx_files(models_dir: Path, recursive: bool = False) -> List[Path]:
    """Return a sorted list of .onnx files under *models_dir*.

    Parameters
    ----------
    models_dir : Path
        Root directory to search.
    recursive : bool
        When True, descend into sub-directories.
    """
    pattern = "**/*.onnx" if recursive else "*.onnx"
    files = sorted(models_dir.glob(pattern))
    if not files:
        logger.warning(
            "No .onnx files found in '%s' (recursive=%s).", models_dir, recursive
        )
    return files


# ---------------------------------------------------------------------------
# 2. ONNX Runtime session
# ---------------------------------------------------------------------------

def create_session(model_path: Path, gpu_index: int) -> ort.InferenceSession:
    """Load an ONNX model and create an ORT session on the specified GPU.

    Raises
    ------
    RuntimeError
        If CUDAExecutionProvider is not available, or if the session silently
        falls back to a CPU provider.
    """
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            f"CUDAExecutionProvider is not available. "
            f"Installed providers: {available}. "
            "Install onnxruntime-gpu and make sure CUDA libraries are on LD_LIBRARY_PATH."
        )

    cuda_opts = {"device_id": gpu_index}
    session = ort.InferenceSession(
        str(model_path),
        providers=[("CUDAExecutionProvider", cuda_opts)],
    )

    # Guard against silent CPU fall-back.
    active = session.get_providers()
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            f"Session for '{model_path.name}' did not use CUDAExecutionProvider "
            f"(active: {active}). Will not run on GPU."
        )

    return session


def make_iobinding(
    session:    ort.InferenceSession,
    inputs:     Dict[str, np.ndarray],
    gpu_index:  int,
) -> ort.IOBinding:
    """Bind inputs and outputs to GPU memory — eliminates GPU→CPU tensor copies."""
    binding = session.io_binding()
    for name, arr in inputs.items():
        ortval = ort.OrtValue.ortvalue_from_numpy(arr, "cuda", gpu_index)
        binding.bind_ortvalue_input(name, ortval)
    for out in session.get_outputs():
        binding.bind_output(out.name, device_type="cuda", device_id=gpu_index)
    return binding


# ---------------------------------------------------------------------------
# 3. Dummy input generation
# ---------------------------------------------------------------------------

def _resolve_dim(dim, position: int, total_dims: int, batch_size: int) -> int:
    """Map one tensor dimension to a concrete positive integer.

    Rules (applied in order)
    ------------------------
    1. Already a positive int  → keep it.
    2. position == 0           → batch_size.
    3. 4-D tensor [B, C, H, W]:
       pos 1 → DEFAULT_CHANNEL
       pos 2 → DEFAULT_HEIGHT
       pos 3 → DEFAULT_WIDTH
    4. 3-D tensor [B, seq, feat]:
       pos 1 → DEFAULT_SEQ_LEN
    5. Fallback                → 1.
    """
    if isinstance(dim, int) and dim > 0:
        return dim

    # Dynamic / symbolic dimension
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
    """Build zero-filled numpy arrays matching the session's input specification.

    Parameters
    ----------
    session : ort.InferenceSession
        A loaded ONNX Runtime session.
    batch_size : int
        Value substituted for dynamic batch dimensions.

    Returns
    -------
    dict
        ``{input_name: np.ndarray}``

    Raises
    ------
    ValueError
        If any required input has a type that cannot be represented as numpy.
    """
    dummy: Dict[str, np.ndarray] = {}

    for inp in session.get_inputs():
        name      = inp.name
        shape     = inp.shape   # list; entries may be int, None, or str
        dtype_str = inp.type

        # --- Resolve dtype ---
        np_dtype = ONNX_DTYPE_MAP.get(dtype_str)
        if np_dtype is None:
            raise ValueError(
                f"Input '{name}' has unsupported type '{dtype_str}'. "
                "Cannot generate a dummy tensor."
            )

        # --- Resolve shape ---
        n_dims   = len(shape)
        resolved = [_resolve_dim(d, i, n_dims, batch_size) for i, d in enumerate(shape)]
        if override_hw is not None and n_dims == 4:
            resolved[2] = override_hw[0]
            resolved[3] = override_hw[1]

        arr = np.zeros(resolved, dtype=np_dtype)
        dummy[name] = arr

        logger.debug("  input '%s': shape=%s  dtype=%s", name, resolved, np_dtype)

    return dummy


# ---------------------------------------------------------------------------
# 4. Energy measurement helpers
# ---------------------------------------------------------------------------

class _PowerSampler:
    """Background thread that polls nvmlDeviceGetPowerUsage() every *dt* s."""

    def __init__(self, handle, dt: float) -> None:
        self._handle     = handle
        self._dt         = dt
        self._stop_event = threading.Event()
        self._lock       = threading.Lock()
        self._timestamps: List[float] = []
        self._power_mw:   List[float] = []
        self._thread = threading.Thread(target=self._run, daemon=True, name="PowerSampler")

    def start(self) -> None:
        self._timestamps.clear()
        self._power_mw.clear()
        self._stop_event.clear()
        self._thread.start()

    def stop(self) -> Tuple[List[float], List[float]]:
        """Signal the thread to stop and return collected (timestamps, power_mw)."""
        self._stop_event.set()
        self._thread.join()
        with self._lock:
            return list(self._timestamps), list(self._power_mw)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            t = time.perf_counter()
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            except pynvml.NVMLError:
                mw = 0
            with self._lock:
                self._timestamps.append(t)
                self._power_mw.append(float(mw))
            time.sleep(self._dt)


class _TempSampler:
    """Background thread that polls nvmlDeviceGetTemperature() every *dt* s."""

    def __init__(self, handle, dt: float) -> None:
        self._handle     = handle
        self._dt         = dt
        self._stop_event = threading.Event()
        self._lock       = threading.Lock()
        self._temps: List[float] = []
        self._thread = threading.Thread(target=self._run, daemon=True, name="TempSampler")

    def start(self) -> None:
        self._temps.clear()
        self._stop_event.clear()
        self._thread.start()

    def stop(self) -> float:
        """Signal the thread to stop and return mean temperature in °C."""
        self._stop_event.set()
        self._thread.join()
        with self._lock:
            return float(np.mean(self._temps)) if self._temps else float("nan")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                t = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                t = float("nan")
            with self._lock:
                self._temps.append(float(t))
            time.sleep(self._dt)


def _trapezoid_energy_joules(timestamps: List[float], power_mw: List[float]) -> float:
    """Trapezoidal integration: power (mW) over time (s) → energy (J)."""
    if len(timestamps) < 2:
        return 0.0
    t = np.asarray(timestamps, dtype=np.float64)
    p = np.asarray(power_mw,   dtype=np.float64) * 1e-3   # mW → W
    return float(np.trapz(p, t))                            # W·s = J


def measure_baseline_power(handle, silence_duration: float, use_energy_counter: bool, dt: float) -> Tuple[float, float]:
    """Measure baseline GPU power and temperature during silence (no inference).

    Uses the hardware energy counter when available (same method as inference),
    otherwise falls back to power sampling.

    Returns (mean_baseline_power_watts, mean_temp_celsius).
    """
    temp_sampler = _TempSampler(handle, dt)
    temp_sampler.start()

    if use_energy_counter:
        energy_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        time.sleep(silence_duration)
        energy_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        energy_j = (energy_end_mj - energy_start_mj) * 1e-3
        baseline_power_w = energy_j / silence_duration
    else:
        power_sampler = _PowerSampler(handle, dt)
        power_sampler.start()
        time.sleep(silence_duration)
        _, power_mw = power_sampler.stop()
        baseline_power_w = float(np.mean(power_mw)) * 1e-3 if power_mw else float("nan")

    avg_temp = temp_sampler.stop()
    return baseline_power_w, avg_temp


def measure_with_energy_counter(
    session:          ort.InferenceSession,
    inputs:           Dict[str, np.ndarray],
    measure_duration: float,
    handle,
    dt:               float,
    iobinding:        Optional[ort.IOBinding] = None,
) -> Tuple[float, float, int, float]:
    """Benchmark using the GPU's hardware energy counter.

    Runs inference repeatedly for *measure_duration* seconds.
    nvmlDeviceGetTotalEnergyConsumption() returns millijoules.

    Returns
    -------
    (total_time_sec, total_energy_joules, num_runs, avg_temp_celsius)
    """
    temp_sampler    = _TempSampler(handle, dt)
    energy_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    temp_sampler.start()

    n  = 0
    t0 = time.perf_counter()
    if iobinding is not None:
        while time.perf_counter() - t0 < measure_duration:
            session.run_with_iobinding(iobinding)
            n += 1
    else:
        while time.perf_counter() - t0 < measure_duration:
            session.run(None, inputs)
            n += 1
    t1 = time.perf_counter()

    energy_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    avg_temp      = temp_sampler.stop()

    total_time_sec      = t1 - t0
    total_energy_joules = (energy_end_mj - energy_start_mj) * 1e-3   # mJ → J
    return total_time_sec, total_energy_joules, n, avg_temp


def measure_with_power_sampling(
    session:          ort.InferenceSession,
    inputs:           Dict[str, np.ndarray],
    measure_duration: float,
    handle,
    dt:               float,
    iobinding:        Optional[ort.IOBinding] = None,
) -> Tuple[float, float, int, float]:
    """Benchmark using background power-sampling (fallback path).

    Runs inference repeatedly for *measure_duration* seconds.
    A daemon thread polls nvmlDeviceGetPowerUsage() every *dt* seconds and
    integrates with the trapezoidal rule to obtain energy.

    Returns
    -------
    (total_time_sec, total_energy_joules, num_runs, avg_temp_celsius)
    """
    power_sampler = _PowerSampler(handle, dt)
    temp_sampler  = _TempSampler(handle, dt)
    power_sampler.start()
    temp_sampler.start()

    n  = 0
    t0 = time.perf_counter()
    if iobinding is not None:
        while time.perf_counter() - t0 < measure_duration:
            session.run_with_iobinding(iobinding)
            n += 1
    else:
        while time.perf_counter() - t0 < measure_duration:
            session.run(None, inputs)
            n += 1
    t1 = time.perf_counter()

    timestamps, power_mw = power_sampler.stop()
    avg_temp             = temp_sampler.stop()

    total_time_sec      = t1 - t0
    total_energy_joules = _trapezoid_energy_joules(timestamps, power_mw)
    return total_time_sec, total_energy_joules, n, avg_temp


# ---------------------------------------------------------------------------
# 5. Per-model processing
# ---------------------------------------------------------------------------

def process_model(
    model_path:         Path,
    args:               argparse.Namespace,
    nvml_handle,
    use_energy_counter: bool,
) -> Optional[Dict]:
    """Load, warm-up, and benchmark one ONNX model.

    Parameters
    ----------
    model_path : Path
        Full path to the .onnx file.
    args : argparse.Namespace
        Parsed CLI arguments.
    nvml_handle :
        NVML device handle.
    use_energy_counter : bool
        True  → use hardware energy counter (primary path).
        False → use power-sampling fallback.

    Returns
    -------
    dict or None
        Result dict on success, None on any failure (error is logged).
    """
    model_name = model_path.name
    logger.info("[%s] Loading session …", model_name)

    # --- Load session ---
    try:
        session = create_session(model_path, args.gpu_index)
    except Exception as exc:
        logger.warning("[%s] Session creation failed: %s", model_name, exc)
        return None

    # --- Build dummy inputs ---
    try:
        override_hw = tuple(args.override_hw) if getattr(args, "override_hw", None) else None
        inputs = make_dummy_inputs(session, batch_size=DEFAULT_BATCH_SIZE, override_hw=override_hw)
    except Exception as exc:
        logger.warning("[%s] Could not build dummy inputs: %s", model_name, exc)
        return None

    if args.verbose:
        for inp_name, arr in inputs.items():
            logger.info("  input '%s': shape=%s  dtype=%s", inp_name, arr.shape, arr.dtype)

    # --- IOBinding (optional) ---
    iobinding = None
    if args.iobinding:
        try:
            iobinding = make_iobinding(session, inputs, args.gpu_index)
            logger.info("[%s] IOBinding enabled — outputs stay on GPU.", model_name)
        except Exception as exc:
            logger.warning("[%s] IOBinding setup failed, falling back to standard run: %s",
                           model_name, exc)
            iobinding = None

    # --- Warm-up ---
    logger.info("[%s] Warm-up (%.1f s) …", model_name, args.warmup_duration)
    try:
        t_warmup = time.perf_counter()
        if iobinding is not None:
            while time.perf_counter() - t_warmup < args.warmup_duration:
                session.run_with_iobinding(iobinding)
        else:
            while time.perf_counter() - t_warmup < args.warmup_duration:
                session.run(None, inputs)
    except Exception as exc:
        logger.warning("[%s] Warm-up failed: %s", model_name, exc)
        return None

    # --- Silence (baseline power + temperature) ---
    logger.info("[%s] Silence (%.1f s) …", model_name, args.silence_duration)
    baseline_power_w, silence_avg_temp = measure_baseline_power(
        nvml_handle, args.silence_duration, use_energy_counter, args.power_sample_dt
    )
    logger.info("[%s] Baseline power: %.3f W  |  silence temp: %.1f °C",
                model_name, baseline_power_w, silence_avg_temp)

    # --- Measured inference ---
    logger.info("[%s] Measuring (%.1f s) …", model_name, args.measure_duration)
    try:
        if use_energy_counter:
            total_time, total_energy, n, inference_avg_temp = measure_with_energy_counter(
                session, inputs, args.measure_duration, nvml_handle, args.power_sample_dt,
                iobinding=iobinding,
            )
        else:
            total_time, total_energy, n, inference_avg_temp = measure_with_power_sampling(
                session, inputs, args.measure_duration, nvml_handle, args.power_sample_dt,
                iobinding=iobinding,
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
        "net_energy/inf=%.6f J  |  net_avgpower/inf=%.3f W  |  baseline=%.3f W  |  "
        "silence_temp=%.1f °C  |  inference_temp=%.1f °C",
        model_name, runtime_perinf, energy_perinf, avgpower_perinf,
        net_energy_perinf, net_avgpower_perinf, baseline_power_w,
        silence_avg_temp, inference_avg_temp,
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
        "silence_avg_temp_c":           silence_avg_temp,
        "inference_avg_temp_c":         inference_avg_temp,
    }


# ---------------------------------------------------------------------------
# 6. CSV output
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
    """Append a single result row to *output_csv*, writing the header only once."""
    write_header = not output_csv.exists()
    df = pd.DataFrame([result], columns=COLUMNS)
    df.to_csv(output_csv, mode="a", index=False, header=write_header)
    logger.info("Appended result for '%s' to '%s'.", result["model_name"], output_csv)


# ---------------------------------------------------------------------------
# 7. CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ONNX models on a CUDA GPU: "
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
        help=(
            "Destination CSV file. "
            "Defaults to output/output_<models_dir_name>_<YYYYMMDD_HHMMSS>.csv"
        ),
    )
    parser.add_argument(
        "--gpu_index", type=int, default=0,
        help="NVIDIA GPU index used for both ORT and NVML.",
    )
    parser.add_argument(
        "--warmup_duration", type=float, default=3.0,
        help="Duration (seconds) of warm-up inference before silence phase.",
    )
    parser.add_argument(
        "--silence_duration", type=float, default=3.0,
        help="Duration (seconds) of GPU silence to measure baseline power.",
    )
    parser.add_argument(
        "--measure_duration", type=float, default=3.0,
        help="Duration (seconds) to run measured inference passes.",
    )
    parser.add_argument(
        "--power_sample_dt", type=float, default=0.01,
        help="Sampling interval (s) for fallback power integration.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Scan sub-directories recursively for .onnx files.",
    )
    parser.add_argument(
        "--iobinding", action="store_true",
        help=(
            "Use ORT IOBinding to keep all output tensors on GPU memory. "
            "Eliminates GPU→CPU PCIe transfer overhead. "
            "Essential for cumulative sub-model benchmarks where many outputs are declared."
        ),
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
# 8. Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # --- Pre-flight checks ---
    if not args.models_dir.is_dir():
        logger.error("--models_dir '%s' is not a directory.", args.models_dir)
        sys.exit(1)

    if args.output_csv is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_csv = Path("output") / f"output_{args.models_dir.name}_{timestamp}.csv"
        logger.info("Output file: %s", args.output_csv)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- NVML initialisation ---
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        logger.error("NVML initialisation failed: %s", exc)
        sys.exit(1)

    nvml_handle        = None
    use_energy_counter = True
    results: List[Dict] = []

    # Ctrl+C: finish the current model, save partial results, then exit cleanly.
    interrupted = threading.Event()

    def _sigint_handler(sig, frame):  # noqa: ANN001
        logger.warning("\nInterrupted by user — saving partial results …")
        interrupted.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
        gpu_name    = pynvml.nvmlDeviceGetName(nvml_handle)
        if isinstance(gpu_name, bytes):          # pynvml < 11 returns bytes
            gpu_name = gpu_name.decode()
        logger.info("GPU %d: %s", args.gpu_index, gpu_name)

        # Probe energy-counter support.
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml_handle)
            use_energy_counter = True
            logger.info("Energy measurement: hardware energy counter (primary path).")
        except pynvml.NVMLError:
            use_energy_counter = False
            logger.info(
                "Energy measurement: power sampling fallback "
                "(interval=%.3f s, trapezoidal integration).",
                args.power_sample_dt,
            )

        # --- Discover ONNX files ---
        model_files = find_onnx_files(args.models_dir, recursive=args.recursive)
        if not model_files:
            logger.error("No models found. Exiting.")
            sys.exit(1)
        logger.info("Found %d ONNX model(s).", len(model_files))

        # --- Process models one by one ---
        total_models  = len(model_files)
        experiment_t0 = time.perf_counter()

        for idx, model_path in enumerate(model_files, start=1):
            if interrupted.is_set():
                logger.info("Stopping early after user interrupt.")
                break

            logger.info("=" * 60)
            logger.info("Model %d / %d : %s", idx, total_models, model_path.name)

            result = process_model(model_path, args, nvml_handle, use_energy_counter)
            if result is not None:
                results.append(result)
                append_result(result, args.output_csv)

            elapsed   = time.perf_counter() - experiment_t0
            avg_per_model = elapsed / idx
            remaining = avg_per_model * (total_models - idx)
            logger.info(
                "Progress: %d / %d done | Elapsed: %.0f s | Estimated remaining: %.0f s (%.1f min)",
                idx, total_models, elapsed, remaining, remaining / 60,
            )

    finally:
        if not results:
            logger.warning("No results to save.")

        pynvml.nvmlShutdown()
        logger.info("NVML shut down. Done — %d model(s) processed.", len(results))


if __name__ == "__main__":
    main()
