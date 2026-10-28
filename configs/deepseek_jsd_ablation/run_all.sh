#!/usr/bin/env bash
# Run the five experiments in order; stop on any training/evaluation failure.
set -euo pipefail
ablation_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ablation_root"
ablation_python=.venv-vopd/bin/python
if [[ ! -x "$ablation_python" ]]; then
  echo 'Run bash scripts/setup_vopd.sh first.' >&2
  exit 1
fi
for ablation_name in 01_entropy_static 02_fkl 03_jsd_fixed_05 04_entropy_mean 05_jsd_linear; do
  "$ablation_python" scripts/run_vopd_ablation.py \
    --config "configs/deepseek_jsd_ablation/${ablation_name}.yaml" "$@"
done
