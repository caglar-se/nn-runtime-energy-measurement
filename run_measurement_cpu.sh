#!/usr/bin/env bash
# run_cpu_energy.sh — wrapper for measure_cpu.py
#
# Usage examples:
#   bash run_cpu_energy.sh --models_dir /path/to/models --rapl_package 0
#   bash run_cpu_energy.sh --models_dir /path/to/models --rapl_package 0 --num_threads 1
#
# Pin the process to cores belonging to the RAPL package you are reading to
# avoid cross-socket energy leakage. Example for a 2-socket system where
# cores 0-15 belong to socket 0 (RAPL package 0):
#   taskset -c 0-15 bash run_cpu_energy.sh --models_dir /path/to/models --rapl_package 0
#
# RAPL read permission:
#   sudo chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj

set -euo pipefail

echo "Running: python measure_cpu.py $*"
exec python measure_cpu.py "$@"
