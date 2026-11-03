#!/usr/bin/env bash
# Run the five core ablations in order; stop on any training/evaluation failure.
# Extra arguments are forwarded to every run, e.g. --dry-run.
set -euo pipefail
ablation_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ablation_root"
ablation_python=.venv-vopd/bin/python
if [[ ! -x "$ablation_python" ]]; then
  echo 'Run bash scripts/setup_vopd.sh first.' >&2
  exit 1
fi
for ablation_name in 01_jsd_q05_noclip 02_mix_q05_noclip 03_jsd_beta05 04_fkl 05_jsd_q05_unnorm; do
  "$ablation_python" scripts/run_vopd_ablation.py \
    --config "configs/deepseek_core_ablation/${ablation_name}.yaml" "$@"
done
