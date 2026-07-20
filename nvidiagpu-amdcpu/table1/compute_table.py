"""
compute_table.py
================
Reads the CSVs produced by run_measurements.sh and prints Table 1:

  NNVC decoder (LOP7 off / on) + isolated LOP7 single-patch measurements,
  with derived overhead, 24× single-patch, and decoder/24× ratios.

Usage
-----
  python examples/table1/compute_table.py \\
      --decoder_csv  examples/table1/results/decoder.csv \\
      --cpu_csvs     examples/table1/results/lop7_cpu_run*.csv \\
      --gpu_csvs     examples/table1/results/lop7_gpu_run*.csv \\
      --patches_per_frame 24 \\
      --output_md    examples/table1/results/table1.md
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _std1(series: pd.Series) -> float:
    """Sample std; returns 0.0 when only one observation."""
    s = series.std(ddof=1)
    return float(s) if not np.isnan(s) else 0.0


def _load_onnx_csvs(paths: list[str], lop7_model: str | None) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if lop7_model:
        df = df[df["model_name"] == lop7_model]
        if df.empty:
            sys.exit(f"Model '{lop7_model}' not found in ONNX CSVs.")
    elif df["model_name"].nunique() > 1:
        names = df["model_name"].unique().tolist()
        sys.exit(
            f"Multiple models found in ONNX CSVs: {names}\n"
            "Use --lop7_model to select one."
        )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate measurement CSVs and print Table 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--decoder_csv", required=True,
                   help="CSV written by run_codec_energy.py")
    p.add_argument("--cpu_csvs", nargs="+", required=True,
                   help="CSV(s) written by measure_cpu.py (one per run)")
    p.add_argument("--gpu_csvs", nargs="+", required=True,
                   help="CSV(s) written by measure_gpu.py / run_measurement_gpu.sh (one per run)")
    p.add_argument("--lop7_model", default=None,
                   help="LOP7 model filename to select (auto if only one model present)")
    p.add_argument("--patches_per_frame", type=int, default=24,
                   help="Number of LOP7 patches per decoded frame")
    p.add_argument("--output_md", default=None,
                   help="If given, also write the table to this Markdown file")
    args = p.parse_args()

    N = args.patches_per_frame

    # -----------------------------------------------------------------------
    # Decoder measurements (condition × stage rows, repeated N_RUNS times)
    # -----------------------------------------------------------------------
    dec = pd.read_csv(args.decoder_csv)
    dec = dec[dec["stage"] == "decoder"].copy()

    # Baseline-subtract to get net energy per decoding pass
    dec["net_energy_j"] = (
        dec["cpu_energy_per_run_j"]
        - dec["cpu_baseline_w"] * dec["runtime_per_run_sec"]
    )
    dec["runtime_ms"] = dec["runtime_per_run_sec"] * 1000.0

    def decoder_stats(condition: str):
        sub = dec[dec["condition"] == condition]
        if sub.empty:
            sys.exit(f"Condition '{condition}' not found in {args.decoder_csv}")
        rt_mean = sub["runtime_ms"].mean()
        rt_std  = _std1(sub["runtime_ms"])
        en_mean = sub["net_energy_j"].mean()
        en_std  = _std1(sub["net_energy_j"])
        return rt_mean, rt_std, en_mean, en_std

    off_rt, off_rt_s, off_en, off_en_s = decoder_stats("lop7_off")
    on_rt,  on_rt_s,  on_en,  on_en_s  = decoder_stats("lop7_on")

    # Overhead = lop7_on − lop7_off; errors add in quadrature
    ovhd_rt   = on_rt - off_rt
    ovhd_rt_s = np.sqrt(on_rt_s**2 + off_rt_s**2)
    ovhd_en   = on_en - off_en
    ovhd_en_s = np.sqrt(on_en_s**2 + off_en_s**2)

    # -----------------------------------------------------------------------
    # Isolated LOP7 — CPU
    # -----------------------------------------------------------------------
    cpu_df = _load_onnx_csvs(args.cpu_csvs, args.lop7_model)
    cpu_rt   = (cpu_df["runtime_perinf_sec"] * 1000.0).mean()
    cpu_rt_s = _std1(cpu_df["runtime_perinf_sec"] * 1000.0)
    cpu_en   = cpu_df["net_energy_perinf_joule"].mean()
    cpu_en_s = _std1(cpu_df["net_energy_perinf_joule"])

    # -----------------------------------------------------------------------
    # Isolated LOP7 — GPU
    # -----------------------------------------------------------------------
    gpu_df = _load_onnx_csvs(args.gpu_csvs, args.lop7_model)
    gpu_rt   = (gpu_df["runtime_perinf_sec"] * 1000.0).mean()
    gpu_rt_s = _std1(gpu_df["runtime_perinf_sec"] * 1000.0)
    gpu_en   = gpu_df["net_energy_perinf_joule"].mean()
    gpu_en_s = _std1(gpu_df["net_energy_perinf_joule"])

    # -----------------------------------------------------------------------
    # Calculated rows — N× single patch
    # -----------------------------------------------------------------------
    cpu24_rt, cpu24_rt_s = N * cpu_rt, N * cpu_rt_s
    cpu24_en, cpu24_en_s = N * cpu_en, N * cpu_en_s
    gpu24_rt, gpu24_rt_s = N * gpu_rt, N * gpu_rt_s
    gpu24_en, gpu24_en_s = N * gpu_en, N * gpu_en_s

    # -----------------------------------------------------------------------
    # Comparison ratios
    # -----------------------------------------------------------------------
    ratio_cpu_rt = ovhd_rt / cpu24_rt
    ratio_cpu_en = ovhd_en / cpu24_en
    ratio_gpu_rt = ovhd_rt / gpu24_rt
    ratio_gpu_en = ovhd_en / gpu24_en

    # -----------------------------------------------------------------------
    # Format helpers
    # -----------------------------------------------------------------------
    def frt(mean, std):
        return f"{mean:.2f} ± {std:.2f}"

    def fen(mean, std):
        return f"{mean:.3f} ± {std:.3f}"

    n_dec = len(dec[dec["condition"] == "lop7_off"])
    n_cpu = len(cpu_df)
    n_gpu = len(gpu_df)
    stats_note = f"decoder {n_dec} run(s), CPU {n_cpu} run(s), GPU {n_gpu} run(s)"
    single_run_warning = (
        "Note: std is 0 for single-run measurements. Use N_RUNS ≥ 5 for reliable ±."
        if (n_dec < 2 or n_cpu < 2 or n_gpu < 2) else ""
    )

    # -----------------------------------------------------------------------
    # Terminal output (plain text table)
    # -----------------------------------------------------------------------
    COL = 28
    RT  = 20
    EN  = 20
    SEP = "-" * (COL + RT + EN)

    def trow(label, rt_m, rt_s, en_m, en_s):
        print(f"  {label:<{COL}}{frt(rt_m, rt_s):>{RT}}{fen(en_m, en_s):>{EN}}")

    def trow_ratio(label, rt_r, en_r):
        print(f"  {label:<{COL}}{f'{rt_r:.2f}×':>{RT}}{f'{en_r:.2f}×':>{EN}}")

    print()
    print(f"  {'Measurement':<{COL}}{'Runtime (ms)':>{RT}}{'Net energy (J)':>{EN}}")
    print(f"  {SEP}")
    print(f"  {'NNVC decoder':}")
    trow("  LOP7 off, CPU",      off_rt,   off_rt_s,   off_en,   off_en_s)
    trow("  LOP7 on, CPU",       on_rt,    on_rt_s,    on_en,    on_en_s)
    trow("  Overhead (on−off)",  ovhd_rt,  ovhd_rt_s,  ovhd_en,  ovhd_en_s)
    print(f"  {SEP}")
    print(f"  {'Isolated LOP7 — CPU':}")
    trow("  Single patch",       cpu_rt,   cpu_rt_s,   cpu_en,   cpu_en_s)
    trow(f"  {N}× single patch", cpu24_rt, cpu24_rt_s, cpu24_en, cpu24_en_s)
    trow_ratio(f"  Overhead / {N}× CPU", ratio_cpu_rt, ratio_cpu_en)
    print(f"  {SEP}")
    print(f"  {'Isolated LOP7 — GPU':}")
    trow("  Single patch",       gpu_rt,   gpu_rt_s,   gpu_en,   gpu_en_s)
    trow(f"  {N}× single patch", gpu24_rt, gpu24_rt_s, gpu24_en, gpu24_en_s)
    trow_ratio(f"  Overhead / {N}× GPU", ratio_gpu_rt, ratio_gpu_en)
    print(f"  {SEP}")
    print()
    print(f"  Statistics: {stats_note}")
    if single_run_warning:
        print(f"  {single_run_warning}")
    print()

    # -----------------------------------------------------------------------
    # Markdown output
    # -----------------------------------------------------------------------
    if args.output_md:
        md_rows = [
            ("NNVC decoder",   "LOP7 off, CPU",          frt(off_rt,   off_rt_s),   fen(off_en,   off_en_s)),
            ("NNVC decoder",   "LOP7 on, CPU",           frt(on_rt,    on_rt_s),    fen(on_en,    on_en_s)),
            ("NNVC decoder",   "Overhead (on − off)",    frt(ovhd_rt,  ovhd_rt_s),  fen(ovhd_en,  ovhd_en_s)),
            ("Isolated LOP7",  "Single patch, CPU",      frt(cpu_rt,   cpu_rt_s),   fen(cpu_en,   cpu_en_s)),
            ("Calculated",     f"{N}× single patch, CPU",frt(cpu24_rt, cpu24_rt_s), fen(cpu24_en, cpu24_en_s)),
            ("Comparison",     f"Overhead / {N}× CPU",   f"{ratio_cpu_rt:.2f}×",     f"{ratio_cpu_en:.2f}×"),
            ("Isolated LOP7",  "Single patch, GPU",      frt(gpu_rt,   gpu_rt_s),   fen(gpu_en,   gpu_en_s)),
            ("Calculated",     f"{N}× single patch, GPU",frt(gpu24_rt, gpu24_rt_s), fen(gpu24_en, gpu24_en_s)),
            ("Comparison",     f"Overhead / {N}× GPU",   f"{ratio_gpu_rt:.2f}×",     f"{ratio_gpu_en:.2f}×"),
        ]

        lines = [
            "# Table 1 — LOP7 Energy Measurement Results",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"Statistics: {stats_note}",
            "",
            "| Measurement | Condition | Runtime (ms) | Net energy (J) |",
            "|---|---|---|---|",
        ]
        for m, cond, rt, en in md_rows:
            lines.append(f"| {m} | {cond} | {rt} | {en} |")

        if single_run_warning:
            lines += ["", f"> {single_run_warning}"]

        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text("\n".join(lines) + "\n")
        print(f"  Markdown saved to: {args.output_md}")
        print()


if __name__ == "__main__":
    main()
