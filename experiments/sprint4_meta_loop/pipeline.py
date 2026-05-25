"""
Sprint 4 — pipeline complet : VG run → inject dynamoscope → analyses.

Étapes :
1. Charge trace JSON depuis vg_traces/
2. POST /api/inject_latents → reçoit coords 3D
3. Lance suite analyses sur cache_key + coords 3D
4. Agrège résultats dans results/

Analyses lancées :
- dynamics_native (Lyapunov, RQA, corr_dim sur 58d)
- topology_native (persistent homology H0/H1 sur 58d)
- dynamics (coords 3D UMAP)
- sindy (équations sparses sur 3D)
- emergence (effective information ladder)
- spectral (DMD, power spectrum)
- causal (Granger / TE / CCM entre dims 3D)

Usage :
    python vg_dynamoscope_pipeline.py --trace vg_mnist_s0_*.json
    python vg_dynamoscope_pipeline.py --all  # toutes traces dans vg_traces/
"""
from __future__ import annotations
import argparse
import glob
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


DYNAMOSCOPE_URL = "http://127.0.0.1:8770"
TRACES_DIR = Path("/Users/zac/claude-code-projects/dynamoscope/cache/vg_traces")
RESULTS_DIR = Path(__file__).parent / "results" / "vg_arch_analysis"


def _post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=180)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:500]
        return {"error": f"HTTP {e.code}: {msg}", "url": url}
    except Exception as e:
        return {"error": str(e), "url": url}


def _post_empty(url: str) -> dict:
    req = urllib.request.Request(url, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=180)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:500]
        return {"error": f"HTTP {e.code}: {msg}", "url": url}
    except Exception as e:
        return {"error": str(e), "url": url}


def run_pipeline(trace_path: Path) -> dict:
    """Charge trace, injecte, lance toutes analyses. Retourne dict aggregate."""
    trace = json.loads(trace_path.read_text())
    cache_key = trace["cache_key"]
    n = trace["n_snapshots"]
    d = trace["dim"]
    print(f"\n=== {trace_path.name} ===")
    print(f"  cache_key={cache_key}  N={n}  D={d}  dataset={trace['dataset']}  seed={trace['seed']}")

    t0 = time.time()
    inject_resp = _post(f"{DYNAMOSCOPE_URL}/api/inject_latents", {
        "cache_key": cache_key,
        "latents": trace["latents"],
        "var_names": trace["var_names"],
        "tags": trace["tags"],
        "steps": trace["steps"],
        "reducer": "umap",
        "meta": trace["meta"],
        "name": f"VG-{trace['dataset']}-s{trace['seed']}",
    })
    if "error" in inject_resp:
        return {"error": inject_resp["error"], "stage": "inject", "trace": trace_path.name}
    coords3d = inject_resp["coords"]
    print(f"  inject OK ({time.time()-t0:.2f}s)  reducer={inject_resp['reducer']}")

    analyses: dict[str, dict] = {}

    # ---- analyses sur cache_key (haute-dim 58) ----
    t = time.time()
    r = _post_empty(f"{DYNAMOSCOPE_URL}/api/dynamics_native/{cache_key}")
    analyses["dynamics_native"] = r
    print(f"  dynamics_native    {time.time()-t:.2f}s  "
          f"{'ok' if 'error' not in r else 'ERR ' + r.get('error', '')[:60]}")

    t = time.time()
    r = _post_empty(f"{DYNAMOSCOPE_URL}/api/topology_native/{cache_key}")
    analyses["topology_native"] = r
    print(f"  topology_native    {time.time()-t:.2f}s  "
          f"{'ok' if 'error' not in r else 'ERR ' + r.get('error', '')[:60]}")

    # ---- analyses sur coords 3D ----
    t = time.time()
    r = _post(f"{DYNAMOSCOPE_URL}/api/dynamics", {"coords": coords3d})
    analyses["dynamics_3d"] = r
    print(f"  dynamics_3d        {time.time()-t:.2f}s  "
          f"{'ok' if 'error' not in r else 'ERR ' + r.get('error', '')[:60]}")

    t = time.time()
    r = _post(f"{DYNAMOSCOPE_URL}/api/sindy", {
        "coords": coords3d, "dt": 1.0, "order": 2, "threshold": 0.05,
        "var_names": ["u1", "u2", "u3"],
    })
    analyses["sindy"] = r
    print(f"  sindy              {time.time()-t:.2f}s  "
          f"{'ok' if 'error' not in r else 'ERR ' + r.get('error', '')[:60]}")

    t = time.time()
    r = _post(f"{DYNAMOSCOPE_URL}/api/emergence", {"coords": coords3d})
    analyses["emergence"] = r
    print(f"  emergence          {time.time()-t:.2f}s  "
          f"{'ok' if 'error' not in r else 'ERR ' + r.get('error', '')[:60]}")

    t = time.time()
    r = _post(f"{DYNAMOSCOPE_URL}/api/spectral", {"coords": coords3d})
    analyses["spectral"] = r
    print(f"  spectral           {time.time()-t:.2f}s  "
          f"{'ok' if 'error' not in r else 'ERR ' + r.get('error', '')[:60]}")

    t = time.time()
    r = _post(f"{DYNAMOSCOPE_URL}/api/causal", {"coords": coords3d})
    analyses["causal"] = r
    print(f"  causal             {time.time()-t:.2f}s  "
          f"{'ok' if 'error' not in r else 'ERR ' + r.get('error', '')[:60]}")

    elapsed = time.time() - t0
    print(f"  TOTAL: {elapsed:.2f}s")

    return {
        "trace_file": trace_path.name,
        "cache_key": cache_key,
        "trace_meta": {
            "dataset": trace["dataset"],
            "seed": trace["seed"],
            "epochs": trace["epochs"],
            "n_snapshots": n,
            "dim": d,
            **trace["meta"],
        },
        "coords_3d": coords3d,
        "tags": trace["tags"],
        "steps": trace["steps"],
        "analyses": analyses,
        "elapsed_sec": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=str, default=None,
                    help="Single trace file (glob) ou nom direct")
    ap.add_argument("--all", action="store_true",
                    help="Run sur toutes traces dans vg_traces/")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    if not args.trace and not args.all:
        ap.error("--trace ou --all requis")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.all:
        files = sorted(TRACES_DIR.glob("vg_*.json"))
    else:
        # Pattern relatif à TRACES_DIR si pas absolu
        pattern = args.trace
        if not Path(pattern).is_absolute():
            pattern = str(TRACES_DIR / pattern)
        files = [Path(p) for p in glob.glob(pattern)]

    if not files:
        print(f"Aucun trace trouvé: {args.trace or 'all'}")
        return

    print(f"Pipeline sur {len(files)} trace(s)")

    summary = []
    for fp in files:
        result = run_pipeline(fp)
        out_file = args.out / f'{result.get("cache_key", fp.stem)}__analyses.json'
        out_file.write_text(json.dumps(result, default=float))
        summary.append({
            "cache_key": result.get("cache_key"),
            "out_file": out_file.name,
            "elapsed": result.get("elapsed_sec"),
            "errors": [k for k, v in result.get("analyses", {}).items() if isinstance(v, dict) and "error" in v],
        })

    print("\n=== summary ===")
    for s in summary:
        err = f"  errors={s['errors']}" if s['errors'] else ""
        print(f"  {s['cache_key']}  {s['elapsed']:.1f}s → {s['out_file']}{err}")

    summary_file = args.out / "summary.json"
    summary_file.write_text(json.dumps(summary, default=float, indent=2))
    print(f"\nresults → {args.out}")


if __name__ == "__main__":
    main()
