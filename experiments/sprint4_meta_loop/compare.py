"""
Sprint 4 — Phase D : agrégation comparative inter-seed.

Question : trajectoires architecture convergent vers même bassin ou divergent ?
Mesures clés (mean ± std sur seeds) :
- Lyapunov (native 58d)
- correlation_dim
- DET / LAM / RR (RQA)
- SINDy R²
- φ_id émergence
- spectral_slope
- DMD rank
- accuracy finale
- mutations acceptées

Sortie : table markdown + JSON aggregate.
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path


RESULTS_DIR = Path(__file__).parent / "results" / "vg_arch_analysis"


def _safe_get(d: dict, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def extract_row(result: dict) -> dict:
    """Extrait métriques-clés d'un fichier d'analyses."""
    a = result.get("analyses", {})
    meta = result.get("trace_meta", {})

    return {
        "cache_key": result.get("cache_key"),
        "seed": meta.get("seed"),
        "dataset": meta.get("dataset"),
        "n_snapshots": meta.get("n_snapshots"),
        "final_acc": meta.get("final_val_acc"),
        "final_params": meta.get("final_params"),
        "final_topology": meta.get("final_topology"),
        "accepted": meta.get("accepted"),
        "rejected": meta.get("rejected"),

        # dynamics_native (58d)
        "lyapunov_58d": _safe_get(a, "dynamics_native", "lyapunov"),
        "corr_dim_58d": _safe_get(a, "dynamics_native", "correlation_dim"),
        "rqa_RR_58d": _safe_get(a, "dynamics_native", "rqa", "RR"),
        "rqa_DET_58d": _safe_get(a, "dynamics_native", "rqa", "DET"),
        "rqa_LAM_58d": _safe_get(a, "dynamics_native", "rqa", "LAM"),
        "convergence_rate_58d": _safe_get(a, "dynamics_native", "convergence_rate"),

        # dynamics_3d (UMAP proj)
        "lyapunov_3d": _safe_get(a, "dynamics_3d", "lyapunov"),
        "corr_dim_3d": _safe_get(a, "dynamics_3d", "correlation_dim"),

        # sindy (3d)
        "sindy_r2": _safe_get(a, "sindy", "r2"),
        "sindy_n_active": _safe_get(a, "sindy", "n_active"),

        # emergence
        "ei_micro": _safe_get(a, "emergence", "ei_micro"),
        "phi_id": _safe_get(a, "emergence", "phi_id_approx"),
        "optimal_scale": _safe_get(a, "emergence", "optimal_scale"),
        "emergence_interp": _safe_get(a, "emergence", "interpretation"),

        # spectral
        "spectral_slope": _safe_get(a, "spectral", "spectral_slope"),
        "dmd_rank": _safe_get(a, "spectral", "dmd", "rank"),
        "dmd_n_modes": _safe_get(a, "spectral", "dmd", "n_modes"),
    }


def aggregate(rows: list[dict]) -> dict:
    """mean ± std sur métriques numériques."""
    if not rows:
        return {}
    numeric_keys = [
        "n_snapshots", "final_acc", "final_params", "accepted", "rejected",
        "lyapunov_58d", "corr_dim_58d", "rqa_RR_58d", "rqa_DET_58d", "rqa_LAM_58d",
        "convergence_rate_58d", "lyapunov_3d", "corr_dim_3d",
        "sindy_r2", "sindy_n_active",
        "ei_micro", "phi_id", "optimal_scale",
        "spectral_slope", "dmd_rank", "dmd_n_modes",
    ]
    agg: dict[str, dict] = {}
    for k in numeric_keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            try:
                vals_f = [float(v) for v in vals]
                agg[k] = {
                    "mean": round(statistics.mean(vals_f), 4),
                    "std": round(statistics.stdev(vals_f), 4) if len(vals_f) > 1 else 0.0,
                    "min": round(min(vals_f), 4),
                    "max": round(max(vals_f), 4),
                    "n": len(vals_f),
                }
            except (TypeError, ValueError):
                pass
    return agg


def md_table(rows: list[dict], cols: list[str]) -> str:
    """Génère table markdown."""
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for r in rows:
        row = []
        for c in cols:
            v = r.get(c)
            if v is None:
                row.append("—")
            elif isinstance(v, float):
                row.append(f"{v:.4f}")
            elif isinstance(v, list):
                row.append(str(v))
            else:
                row.append(str(v))
        body.append("| " + " | ".join(row) + " |")
    return "\n".join([header, sep] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "comparison.md")
    ap.add_argument("--out-json", type=Path, default=RESULTS_DIR / "comparison.json")
    args = ap.parse_args()

    files = sorted(args.in_dir.glob("*__analyses.json"))
    if not files:
        print(f"No analyses in {args.in_dir}")
        return

    rows = []
    for fp in files:
        result = json.loads(fp.read_text())
        rows.append(extract_row(result))

    rows.sort(key=lambda r: r.get("seed", 0))

    agg = aggregate(rows)

    cols_main = ["seed", "final_acc", "final_topology", "accepted", "rejected",
                 "lyapunov_58d", "corr_dim_58d", "rqa_DET_58d", "rqa_LAM_58d",
                 "sindy_r2", "phi_id", "spectral_slope", "dmd_rank"]
    table = md_table(rows, cols_main)

    md = []
    md.append(f"# VG architecture-as-trajectory — comparaison inter-seed")
    md.append(f"\n**N runs** : {len(rows)}  · **dataset** : {rows[0].get('dataset')}\n")
    md.append("## Per-seed metrics\n")
    md.append(table)
    md.append("\n## Aggregate (mean ± std)\n")
    md.append("| metric | mean | std | min | max | n |")
    md.append("|---|---|---|---|---|---|")
    for k, v in agg.items():
        md.append(f"| {k} | {v['mean']} | {v['std']} | {v['min']} | {v['max']} | {v['n']} |")

    # Verdict
    md.append("\n## Verdict\n")
    if "final_topology" in cols_main:
        topos = set(str(r.get("final_topology")) for r in rows)
        md.append(f"- **Topologies finales** : {len(topos)} distinctes — `{topos}`")
    acc_std = agg.get("final_acc", {}).get("std", 0)
    md.append(f"- **Acc std** : {acc_std} (basse = convergence)")
    if "sindy_r2" in agg:
        r2_mean = agg["sindy_r2"]["mean"]
        md.append(f"- **SINDy R² moyen** : {r2_mean} (>0.8 = dynamics modélisables sparsement)")
    if "phi_id" in agg:
        phi = agg["phi_id"]["mean"]
        md.append(f"- **φ_id moyen** : {phi} (>0 = émergence détectée)")
    if "lyapunov_58d" in agg:
        ly = agg["lyapunov_58d"]["mean"]
        md.append(f"- **Lyapunov 58d moyen** : {ly} (≈0 = stable contractif, >0 = chaotique)")

    args.out.write_text("\n".join(md))
    args.out_json.write_text(json.dumps({"rows": rows, "aggregate": agg}, default=float, indent=2))

    print("\n".join(md))
    print(f"\n→ {args.out}")
    print(f"→ {args.out_json}")


if __name__ == "__main__":
    main()
