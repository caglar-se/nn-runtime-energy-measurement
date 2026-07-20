#!/usr/bin/env bash
# run_measurement_cpu_macos.sh — wrapper for measure_cpu_macos.py
#
# Usage examples:
#   bash run_measurement_cpu_macos.sh --models_dir /path/to/models
#   bash run_measurement_cpu_macos.sh --models_dir /path/to/models --num_threads 1
#
# Passwordless sudo for powermetrics:
#   sudo sh -c 'echo "'"$USER"' ALL=(ALL) NOPASSWD: /usr/bin/powermetrics" \
#     > /etc/sudoers.d/powermetrics && chmod 440 /etc/sudoers.d/powermetrics'

set -euo pipefail

echo "Running: python measure_cpu_macos.py $*"
exec python measure_cpu_macos.py "$@"
