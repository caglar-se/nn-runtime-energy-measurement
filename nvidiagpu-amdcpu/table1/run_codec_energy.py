"""
run_codec_energy.py
===================
Measures per-invocation runtime and energy for the VTM NNVC encoder and decoder
with and without the LOP7 neural-network loop filter (NnlfOption).

Measurement protocol  (mirrors analyse_nn_energy.py)
-----------------------------------------------------
For each stage (encoder, decoder):
  1. Warm-up   : run subprocess in a loop for --warmup_duration seconds
  2. Silence   : idle for --silence_duration seconds  →  baseline power
  3. Measurement: run subprocess in a loop for --measure_duration seconds
                  bracket the whole window with RAPL / NVML counters
                  per-invocation energy = total_energy / num_runs

For fast processes (decoder, ~0.2–0.5 s): many runs accumulate in 3 s → good SNR.
For slow processes (encoder, ~23 s):  the loop runs exactly once and already
exceeds the measurement window — single-run measurement is reliable.

Usage
-----
  cd /workspace/NN_Energy
  taskset -c 40-49 python3 jvet-jul26/run_codec_energy.py \\
      --vtm_dir  jvet-jul26/VVCSoftware_VTM \\
      --input_yuv jvet-jul26/VVCSoftware_VTM/CTC_Image/yuv_output_ctc/kodim19_512x768_8bit_420.yuv \\
      --base_cfg cfg/encoder_intra_nnvc.cfg \\
      --seq_cfg  cfg/CTC_JPEGAI/kodim19_512x768_8bit_420.cfg \\
      --frames 1 --qp 42 --runs 5 \\
      --rapl_package 1 --gpu_index 4 \\
      --output_csv jvet-jul26/results/codec_energy_image_s1.csv

Output CSV columns
------------------
date_time, condition, stage,
runtime_per_run_sec, cpu_energy_per_run_j, gpu_energy_per_run_j,
cpu_baseline_w, gpu_baseline_w,
num_runs, total_time_sec,
frames, qp
"""

import argparse
import csv
import logging
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RAPL (CPU package energy)
# ---------------------------------------------------------------------------

RAPL_BASE = Path("/sys/class/powercap/intel-rapl")


def _rapl_path(package: int) -> Path:
    return RAPL_BASE / f"intel-rapl:{package}"


def read_rapl_uj(package: int) -> int:
    return int((_rapl_path(package) / "energy_uj").read_text().strip())


def rapl_max_uj(package: int) -> int:
    return int((_rapl_path(package) / "max_energy_range_uj").read_text().strip())


def rapl_delta_j(start_uj: int, end_uj: int, max_uj: int) -> float:
    if end_uj >= start_uj:
        return (end_uj - start_uj) * 1e-6
    return (max_uj - start_uj + end_uj) * 1e-6


def check_rapl(package: int) -> bool:
    path = _rapl_path(package)
    if not path.exists():
        logger.warning("RAPL package %d not found — CPU energy disabled.", package)
        return False
    try:
        read_rapl_uj(package)
        return True
    except PermissionError:
        logger.warning(
            "Cannot read RAPL package %d — CPU energy disabled. "
            "Fix: sudo chmod o+r %s/energy_uj", package, path
        )
        return False


# ---------------------------------------------------------------------------
# powermetrics (macOS Apple Silicon CPU energy)
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
        logger.debug("powermetrics: %d samples, avg %.1f mW", len(self._samples_mw), avg_w * 1000)
        return avg_w * elapsed_sec

    def stop_and_get_avg_w(self, elapsed_sec: float) -> float:
        energy_j = self.stop_and_get_energy_j(elapsed_sec)
        if elapsed_sec > 0 and not (energy_j != energy_j):  # not nan
            return energy_j / elapsed_sec
        return float("nan")


def check_powermetrics() -> bool:
    try:
        r = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"],
            capture_output=True, timeout=5,
        )
        if r.returncode == 0:
            return True
        import getpass
        logger.warning(
            "powermetrics check failed (exit %d) — CPU energy disabled. "
            "Fix: sudo sh -c 'echo \"%s ALL=(ALL) NOPASSWD: /usr/bin/powermetrics\" "
            "> /etc/sudoers.d/powermetrics && chmod 440 /etc/sudoers.d/powermetrics'",
            r.returncode, getpass.getuser(),
        )
        return False
    except Exception as e:
        logger.warning("powermetrics check failed: %s — CPU energy disabled.", e)
        return False


# ---------------------------------------------------------------------------
# NVML (GPU energy)
# ---------------------------------------------------------------------------

def _try_import_pynvml():
    try:
        import pynvml
        return pynvml
    except ImportError:
        return None


def init_nvml(gpu_index: int):
    pynvml = _try_import_pynvml()
    if pynvml is None:
        logger.warning("pynvml not installed — GPU energy disabled.")
        return None, None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        logger.info("GPU %d: %s", gpu_index, name)
        return pynvml, handle
    except Exception as exc:
        logger.warning("NVML init failed: %s — GPU energy disabled.", exc)
        return None, None


def read_nvml_mj(pynvml, handle) -> Optional[int]:
    if pynvml is None:
        return None
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Baseline measurement
# ---------------------------------------------------------------------------

def measure_baseline(
    rapl_ok: bool, rapl_package: int, rapl_max: int,
    pynvml, nvml_handle, silence: float,
    pm_ok: bool = False,
) -> Tuple[float, float]:
    """Sleep for silence seconds, return (cpu_baseline_w, gpu_baseline_w)."""
    pm = PowermetricsReader() if pm_ok else None
    if pm:
        pm.start()

    cpu_s = read_rapl_uj(rapl_package) if rapl_ok else 0
    gpu_s = read_nvml_mj(pynvml, nvml_handle)
    t0 = time.perf_counter()
    time.sleep(silence)
    elapsed = time.perf_counter() - t0
    cpu_e = read_rapl_uj(rapl_package) if rapl_ok else 0
    gpu_e = read_nvml_mj(pynvml, nvml_handle)

    if pm:
        cpu_w = pm.stop_and_get_avg_w(elapsed)
    elif rapl_ok and elapsed > 0:
        cpu_w = rapl_delta_j(cpu_s, cpu_e, rapl_max) / elapsed
    else:
        cpu_w = float("nan")

    gpu_w = float("nan")
    if gpu_s is not None and gpu_e is not None and elapsed > 0:
        gpu_w = (gpu_e - gpu_s) * 1e-3 / elapsed
    return cpu_w, gpu_w


# ---------------------------------------------------------------------------
# Core: run subprocess in a loop for a fixed duration
# ---------------------------------------------------------------------------

def _run_cmd(cmd, run_dir) -> bool:
    """Run cmd once. Return True on success."""
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(run_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(
            "Process failed (code %d)\nSTDERR: %s",
            result.returncode, result.stderr[-1000:],
        )
        return False
    return True


def warmup_loop(cmd, run_dir: Path, warmup_duration: float, label: str) -> int:
    """Run cmd repeatedly for warmup_duration seconds. Returns number of runs."""
    logger.info("[%s] Warm-up (%.1f s) …", label, warmup_duration)
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warmup_duration:
        if not _run_cmd(cmd, run_dir):
            break
        n += 1
    logger.info("[%s] Warm-up done: %d run(s) in %.1f s.", label, n, time.perf_counter() - t0)
    return n


def measure_loop(
    cmd, run_dir: Path, measure_duration: float,
    rapl_ok: bool, rapl_package: int, rapl_max: int,
    pynvml, nvml_handle, label: str,
    pm_ok: bool = False,
) -> Tuple[float, float, float, int]:
    """
    Run cmd repeatedly for measure_duration seconds, bracketed by energy counters.

    Returns (total_wall_time_sec, cpu_energy_j, gpu_energy_j, num_runs).
    Energy values are for the full window; divide by num_runs for per-invocation cost.
    """
    logger.info("[%s] Measuring (%.1f s) …", label, measure_duration)

    pm = PowermetricsReader() if pm_ok else None
    if pm:
        pm.start()

    cpu_s = read_rapl_uj(rapl_package) if rapl_ok else 0
    gpu_s = read_nvml_mj(pynvml, nvml_handle)

    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < measure_duration:
        if not _run_cmd(cmd, run_dir):
            break
        n += 1
    t1 = time.perf_counter()

    cpu_e = read_rapl_uj(rapl_package) if rapl_ok else 0
    gpu_e = read_nvml_mj(pynvml, nvml_handle)

    total_time = t1 - t0

    if pm:
        cpu_j = pm.stop_and_get_energy_j(total_time)
    elif rapl_ok:
        cpu_j = rapl_delta_j(cpu_s, cpu_e, rapl_max)
    else:
        cpu_j = float("nan")

    gpu_j = float("nan")
    if gpu_s is not None and gpu_e is not None:
        gpu_j = (gpu_e - gpu_s) * 1e-3

    if n == 0:
        logger.error("[%s] No successful runs in measurement window.", label)
        return total_time, float("nan"), float("nan"), 0

    logger.info(
        "[%s] %d run(s) in %.2f s | cpu_total=%.2f J | gpu_total=%.2f J "
        "| cpu/run=%.4f J | runtime/run=%.4f s",
        label, n, total_time, cpu_j, gpu_j,
        cpu_j / n if n > 0 else float("nan"),
        total_time / n,
    )
    return total_time, cpu_j, gpu_j, n


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

COLUMNS = [
    "date_time", "image", "condition", "stage",
    "runtime_per_run_sec", "cpu_energy_per_run_j", "gpu_energy_per_run_j",
    "cpu_baseline_w", "gpu_baseline_w",
    "num_runs", "total_time_sec",
    "frames", "qp",
]


def append_row(row: dict, csv_path: Path) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# One encode + decode pass
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    nnlf_option: int,
    args: argparse.Namespace,
    rapl_ok: bool,
    rapl_max: int,
    pynvml,
    nvml_handle,
    csv_path: Path,
    pm_ok: bool = False,
) -> None:
    vtm_dir  = Path(args.vtm_dir).resolve()
    encoder  = vtm_dir / "bin" / "EncoderAppStatic"
    decoder  = vtm_dir / "bin" / "DecoderAppStatic"
    base_cfg = vtm_dir / args.base_cfg
    seq_cfg  = vtm_dir / args.seq_cfg
    input_yuv = Path(args.input_yuv).resolve()

    for p in [encoder, decoder, base_cfg, seq_cfg, input_yuv]:
        if not p.exists():
            logger.error("Required path not found: %s", p)
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Condition: %s  (NnlfOption=%d)", condition, nnlf_option)
    logger.info("=" * 60)

    # Bitstream lives in a temp dir for the duration of this condition.
    with tempfile.TemporaryDirectory(prefix=f"vtm_{condition}_") as tmpdir:
        tmp      = Path(tmpdir)
        bitstream = tmp / "str.bin"
        recon    = tmp / "rec.yuv"
        dec_yuv  = tmp / "dec.yuv"

        # Run from vtm_dir so relative model paths in configs resolve correctly.
        run_dir = vtm_dir

        enc_cmd = [
            encoder,
            "-c", base_cfg, "-c", seq_cfg,
            f"--NnlfOption={nnlf_option}",
            "-i", input_yuv,
            "-b", bitstream, "-o", recon,
            "-f", str(args.frames), "-q", str(args.qp),
        ]
        dec_cmd = [decoder, "-b", bitstream, "-o", dec_yuv]
        if nnlf_option > 0:
            dec_cmd += [f"--NnlfModelName={args.nnlf_model}"]

        # ----------------------------------------------------------------
        # ENCODER
        # ----------------------------------------------------------------
        enc_label = f"{condition}/encoder"

        # Warm-up (also produces the bitstream needed by decoder warmup).
        warmup_loop(enc_cmd, run_dir, args.warmup_duration, enc_label)

        # Baseline.
        logger.info("[%s] Silence (%.1f s) …", enc_label, args.silence_duration)
        enc_cpu_base, enc_gpu_base = measure_baseline(
            rapl_ok, args.rapl_package, rapl_max,
            pynvml, nvml_handle, args.silence_duration, pm_ok=pm_ok,
        )
        logger.info("[%s] Baseline: cpu=%.3f W  gpu=%.3f W",
                    enc_label, enc_cpu_base, enc_gpu_base)

        # Measurement loop.
        enc_total_t, enc_cpu_j, enc_gpu_j, enc_n = measure_loop(
            enc_cmd, run_dir, args.measure_duration,
            rapl_ok, args.rapl_package, rapl_max,
            pynvml, nvml_handle, enc_label, pm_ok=pm_ok,
        )

        if enc_n > 0:
            append_row({
                "date_time":              time.strftime("%Y-%m-%d %H:%M:%S"),
                "image":                  Path(args.input_yuv).stem,
                "condition":              condition,
                "stage":                  "encoder",
                "runtime_per_run_sec":    enc_total_t / enc_n,
                "cpu_energy_per_run_j":   enc_cpu_j   / enc_n,
                "gpu_energy_per_run_j":   enc_gpu_j   / enc_n,
                "cpu_baseline_w":         enc_cpu_base,
                "gpu_baseline_w":         enc_gpu_base,
                "num_runs":               enc_n,
                "total_time_sec":         enc_total_t,
                "frames":                 args.frames,
                "qp":                     args.qp,
            }, csv_path)

        # ----------------------------------------------------------------
        # DECODER
        # ----------------------------------------------------------------
        dec_label = f"{condition}/decoder"

        # Warm-up decoder (bitstream already on disk from encoder warm-up).
        warmup_loop(dec_cmd, run_dir, args.warmup_duration, dec_label)

        # Baseline.
        logger.info("[%s] Silence (%.1f s) …", dec_label, args.silence_duration)
        dec_cpu_base, dec_gpu_base = measure_baseline(
            rapl_ok, args.rapl_package, rapl_max,
            pynvml, nvml_handle, args.silence_duration, pm_ok=pm_ok,
        )
        logger.info("[%s] Baseline: cpu=%.3f W  gpu=%.3f W",
                    dec_label, dec_cpu_base, dec_gpu_base)

        # Measurement loop.
        dec_total_t, dec_cpu_j, dec_gpu_j, dec_n = measure_loop(
            dec_cmd, run_dir, args.measure_duration,
            rapl_ok, args.rapl_package, rapl_max,
            pynvml, nvml_handle, dec_label, pm_ok=pm_ok,
        )

        if dec_n > 0:
            append_row({
                "date_time":              time.strftime("%Y-%m-%d %H:%M:%S"),
                "image":                  Path(args.input_yuv).stem,
                "condition":              condition,
                "stage":                  "decoder",
                "runtime_per_run_sec":    dec_total_t / dec_n,
                "cpu_energy_per_run_j":   dec_cpu_j   / dec_n,
                "gpu_energy_per_run_j":   dec_gpu_j   / dec_n,
                "cpu_baseline_w":         dec_cpu_base,
                "gpu_baseline_w":         dec_gpu_base,
                "num_runs":               dec_n,
                "total_time_sec":         dec_total_t,
                "frames":                 args.frames,
                "qp":                     args.qp,
            }, csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark VTM NNVC encoder+decoder energy with/without LOP7. "
            "Mirrors the warm-up / silence / measure-loop protocol of analyse_nn_energy.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vtm_dir",     default="jvet-jul26/VVCSoftware_VTM")
    p.add_argument("--input_yuv",
                   default="jvet-jul26/VVCSoftware_VTM/CTC_Image/yuv_output_ctc/kodim19_512x768_8bit_420.yuv")
    p.add_argument("--base_cfg",    default="cfg/encoder_intra_nnvc.cfg")
    p.add_argument("--seq_cfg",     default="cfg/CTC_JPEGAI/kodim19_512x768_8bit_420.cfg")
    p.add_argument("--frames",      type=int,   default=1)
    p.add_argument("--qp",          type=int,   default=42)
    p.add_argument("--runs",        type=int,   default=5,
                   help="Number of outer repetitions (each is a full warmup+silence+measure cycle).")
    p.add_argument("--warmup_duration",  type=float, default=3.0,
                   help="Seconds to run subprocess loop as warm-up (discarded).")
    p.add_argument("--silence_duration", type=float, default=3.0,
                   help="Seconds of idle silence to measure baseline power.")
    p.add_argument("--measure_duration", type=float, default=3.0,
                   help="Seconds to run subprocess loop for energy measurement.")
    p.add_argument("--rapl_package", type=int, default=1)
    p.add_argument("--gpu_index",    type=int, default=4)
    p.add_argument("--energy_backend", default="auto",
                   choices=["auto", "rapl", "powermetrics", "none"],
                   help="CPU energy backend. 'auto' picks rapl on Linux, powermetrics on macOS.")
    p.add_argument("--output_csv",   type=Path, default=None)
    p.add_argument("--nnlf_model",   default="models/nnlf_lop7_model_int16.sadl",
                   help="SADL model path passed to --NnlfModelName (relative to vtm_dir).")
    p.add_argument("--conditions",   nargs="+", default=["lop7_off", "lop7_on"],
                   choices=["lop7_off", "lop7_on"])
    return p.parse_args()


CONDITION_NNLF = {"lop7_on": 3, "lop7_off": 0}


def main() -> None:
    args = parse_args()

    if args.output_csv is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output_csv = Path("jvet-jul26/results") / f"codec_energy_{ts}.csv"
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Output CSV: %s", args.output_csv)

    backend = args.energy_backend
    if backend == "auto":
        backend = "powermetrics" if platform.system() == "Darwin" else "rapl"
    logger.info("Energy backend: %s", backend)

    if backend == "rapl":
        rapl_ok = check_rapl(args.rapl_package)
        rapl_max = rapl_max_uj(args.rapl_package) if rapl_ok else 1
        pm_ok = False
    elif backend == "powermetrics":
        rapl_ok = False
        rapl_max = 1
        pm_ok = check_powermetrics()
    else:  # none
        rapl_ok = False
        rapl_max = 1
        pm_ok = False

    if args.gpu_index >= 0:
        pynvml, nvml_handle = init_nvml(args.gpu_index)
    else:
        pynvml, nvml_handle = None, None

    try:
        for run_idx in range(1, args.runs + 1):
            logger.info("### Outer run %d / %d ###", run_idx, args.runs)
            for condition in args.conditions:
                run_condition(
                    condition, CONDITION_NNLF[condition], args,
                    rapl_ok, rapl_max, pynvml, nvml_handle,
                    args.output_csv, pm_ok=pm_ok,
                )
    finally:
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    logger.info("Done. Results: %s", args.output_csv)


if __name__ == "__main__":
    main()
