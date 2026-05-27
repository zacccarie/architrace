"""autoresearch wrapper for architrace experiments.

Usage:
    python autoresearch/wrapper.py sprint3_1     # bench extended (corr_dim delta)
    python autoresearch/wrapper.py sprint17b     # pretrain scratch
    python autoresearch/wrapper.py sprint1       # predictor swap

Wrapper rôle :
1. Lance l'expérience sélectionnée.
2. Parse JSON résultats produit par l'expérience.
3. Imprime footer parseable pour autoresearch loop.

Footer attendu :
    ---
    delta_corr_dim:      0.234
    vg_tiny_delta:       0.198
    resnet50_rand_delta: 0.301
    training_seconds:    245.3
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO_ROOT / "experiments"
RESULTS = REPO_ROOT / "results"


def runSprint31() -> dict:
    """Sprint 3.1 extended bench — encoders vs corr_dim ground truth."""
    script = EXPERIMENTS / "sprint3_1_extended_bench" / "bench_extended.py"
    if not script.exists():
        raise FileNotFoundError(f"Sprint 3.1 script absent : {script}")
    out_dir = RESULTS / "sprint3_1_autoresearch"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(
        [sys.executable, str(script), "--out", str(out_dir)],
        check=True, cwd=REPO_ROOT,
    )
    elapsed = time.time() - t0
    results_file = out_dir / "bench_extended_results.json"
    with results_file.open() as f:
        data = json.load(f)
    rows = data["rows"]
    gt = data["gt"]
    # Delta corr_dim global et par encoder
    deltas = {}
    for enc in ["vg_tiny", "resnet50_random", "resnet50_pretrained"]:
        per_row = []
        for r in rows:
            if r["encoder"] == enc:
                gt_list = gt[r["system"]]
                # Match seed index — bench_extended stocke par index seed
                idx = r.get("seed_idx", 0) if "seed_idx" in r else 0
                if idx < len(gt_list):
                    per_row.append(abs(r["corr_dim"] - gt_list[idx]["corr_dim"]))
        deltas[enc] = float(np.mean(per_row)) if per_row else float("nan")
    # Métrique décisionnelle : moyenne vg_tiny (notre encoder)
    return {
        "delta_corr_dim": deltas["vg_tiny"],
        "vg_tiny_delta": deltas["vg_tiny"],
        "resnet50_rand_delta": deltas["resnet50_random"],
        "resnet50_pre_delta": deltas["resnet50_pretrained"],
        "training_seconds": round(elapsed, 1),
    }


def runSprint17b() -> dict:
    """Sprint 17b pretrain scratch — NON PRIORITAIRE Q2 2026.

    Voir autoresearch/CONTEXT.md : architrace = read for context, don't iterate.
    Si vraiment nécessaire, brancher manuellement après revue scientifique.
    """
    raise RuntimeError(
        "Sprint 17b out of autoresearch scope (architrace = stable, paper-finalize phase). "
        "Cf. autoresearch/CONTEXT.md."
    )


def runSprint1() -> dict:
    """Sprint 1 predictor swap — NON PRIORITAIRE Q2 2026. Voir CONTEXT.md."""
    raise RuntimeError(
        "Sprint 1 out of autoresearch scope (architrace = stable, paper-finalize phase). "
        "Cf. autoresearch/CONTEXT.md."
    )


DISPATCH = {
    "sprint3_1": runSprint31,
    # Désactivés intentionnellement — voir CONTEXT.md.
    # "sprint17b": runSprint17b,
    # "sprint1": runSprint1,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage : {sys.argv[0]} <{'|'.join(DISPATCH)}>", file=sys.stderr)
        sys.exit(2)
    name = sys.argv[1]
    fn = DISPATCH.get(name)
    if fn is None:
        print(f"Expérience inconnue : {name}. Dispos : {list(DISPATCH)}", file=sys.stderr)
        sys.exit(2)
    metrics = fn()
    print("\n---")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
