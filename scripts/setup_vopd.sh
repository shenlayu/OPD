#!/usr/bin/env bash
# Recreate the environment on the destination machine; never copy a venv.
set -euo pipefail
vopd_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$vopd_root"
vopd_bootstrap=${VOPD_BOOTSTRAP_PYTHON:-python3.12}
"$vopd_bootstrap" - <<'PY'
import platform
import sys
if sys.version_info[:2] != (3, 12) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('The pinned vOPD environment requires Python 3.12 on Linux x86_64.')
PY
"$vopd_bootstrap" -m venv .venv-vopd
.venv-vopd/bin/python -m pip install -r requirements-vopd.lock.txt
.venv-vopd/bin/python -m pip check
echo 'vOPD environment ready: .venv-vopd/bin/python'
