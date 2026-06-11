#!/usr/bin/env bash
# One-shot runner for the identifiability framework (v2).
# Usage:  bash notebooks/v2/run_all.sh
# Run from the repository root.
set -e

echo "==> 0. dependencies (numpy/scipy/pandas/matplotlib)"
python3 -c "import numpy,scipy,pandas,matplotlib" 2>/dev/null \
  || pip install -q numpy scipy pandas matplotlib

echo "==> 1. (re)generate synthetic benchmark data -> data/matlab_inputs/"
python3 -c "import sys;sys.path.insert(0,'src');\
from synthetic.export_for_matlab import export_all_scenarios as e;e('data/matlab_inputs')"

echo "==> 2. honest single-pipeline benchmark (synthetic, all 5 scenarios)"
python3 notebooks/v2/honest_benchmark.py

echo "==> 3. Siheung field application (needs data/field/SH*.txt present locally)"
if ls data/field/SH*.txt >/dev/null 2>&1; then
  python3 notebooks/v2/field_identifiability.py
else
  echo "    (skipped: no data/field/SH*.txt — field data is kept local)"
fi

echo "==> 4. regenerate figures -> notebooks/figures/"
python3 notebooks/v2/make_identifiability_figures.py

echo "==> 5. tests"
python3 -m pytest -q

echo "==> done."
