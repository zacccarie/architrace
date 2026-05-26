"""
architrace — fusion VerifyGrowNet × Dynamoscope.

Modules :
    snapshot     extracteur features architecture (58D vec)
    bridge       client HTTP dynamoscope (inject + analyses)
    trajectory   API haut-niveau (run + analyse)
    predictor_vg option 1 : VG-based predictor pour rollout dynamoscope
"""
from __future__ import annotations

__version__ = "0.1.0"

# Configuration paths (override via env var ou config)
import os
from pathlib import Path

ADAPTIVE_NN_PATH = Path(os.environ.get(
    "ARCHITRACE_ADAPTIVE_NN",
    "/Users/zac/claude-code-projects/adaptive-nn",
))
DYNAMOSCOPE_PATH = Path(os.environ.get(
    "ARCHITRACE_DYNAMOSCOPE",
    "/Users/zac/claude-code-projects/dynamoscope",
))
DYNAMOSCOPE_URL = os.environ.get(
    "ARCHITRACE_DYNAMOSCOPE_URL",
    "http://127.0.0.1:8770",
)

# Ajout sys.path pour imports upstream
# adaptive-nn restructuré 2026-05-26 : core/ contient verifygrow_*, adaptive_nn,
# events, ewc, icarl, vg_ncm, vg_continual, arch_snapshot, server
import sys as _sys
_paths = [
    ADAPTIVE_NN_PATH,
    ADAPTIVE_NN_PATH / 'core',
    DYNAMOSCOPE_PATH,
]
for _p in _paths:
    if _p.exists() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
