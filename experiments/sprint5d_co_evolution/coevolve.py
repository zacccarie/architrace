"""
Sprint 5D — co-évolution observer mutable × observed mutable.

Setup :
- Observed = VG-MLP qui pousse via dynamoscope φ_id trigger (Sprint 2/5B)
- Observer = MutableVGCNNObserver qui pousse via SSVG plateau sur SA trace

Boucle :
    Pour chaque batch training :
        - VG-MLP forward + grad step
        - Render arch state image
        - Observer SSVG step (apprend à prédire prochain état)
        - Observer encode image → push trace dynamoscope
        - Si φ_id drop → mutation VG-MLP
        - Si SSVG plateau → mutation observer

Mesure :
- N mutations VG-MLP
- N mutations observer
- Corrélation timing mutations (phase-lock ?)
- Acc finale VG-MLP
- Topology finale des deux
"""
from __future__ import annotations
import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import architrace  # noqa
from architrace.arch_to_image import render_arch_state
from architrace.bridge import DynamoscopeClient
from architrace.mutable_observer import MutableVGCNNObserver
from verifygrow_nn import VerifyGrowNet, bottleneck_layer  # type: ignore


def load_mnist(n_train=6000, n_val=1500):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


@torch.no_grad()
def eval_acc(model, loader, max_batches=10) -> float:
    model.eval()
    correct = 0; total = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    model.train()
    return correct / max(1, total)


def run_coevolution(seed: int, client: DynamoscopeClient, verbose: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    # OBSERVED = VG-MLP
    observed = VerifyGrowNet(in_dim=784, out_dim=10, init_hidden=20, init_depth=2)
    opt_obs = torch.optim.Adam(observed.parameters(), lr=1e-3)

    # OBSERVER = MutableVGCNNObserver (random init)
    observer = MutableVGCNNObserver(channels=(4, 8), input_size=64)
    opt_observer = torch.optim.AdamW(observer.parameters(), lr=1e-3, weight_decay=1e-4)

    # Buffers
    encoded_buf: list[np.ndarray] = []  # observer's view of observed
    observer_ssvg_buf: deque = deque(maxlen=40)
    phi_history: list[float] = []
    phi_ema = None
    phi_baseline = None

    observed_mutations: list[dict] = []
    observer_mutations: list[dict] = []
    pending_observed: dict | None = None

    prev_img: np.ndarray | None = None
    step = 0
    last_phi_eval = -1
    last_observed_mut = -10000
    last_observer_grow_try = -10000

    if verbose:
        print(f'[init] observed: {observed.topology()} {observed.num_params()}p · '
              f'observer: {observer.topology()} {observer.num_params()}p')

    for ep in range(10):
        for x, y in tr:
            # --- Observed training step ---
            observed.train()
            opt_obs.zero_grad()
            logits = observed(x)
            task_loss = F.cross_entropy(logits, y)
            task_loss.backward()
            opt_obs.step()

            # --- Render arch image ---
            curr_img = render_arch_state(observed, image_size=64)

            # --- Observer SSVG step ---
            if prev_img is not None and step > 0:
                ssvg_loss = observer.train_ssvg_step(prev_img, curr_img, opt_observer)
                observer_ssvg_buf.append(ssvg_loss)

            # --- Observer encode → trace ---
            if step % 3 == 0:
                z = observer.encode(curr_img[None, ...])[0]
                encoded_buf.append(z)
                if len(encoded_buf) > 200:
                    encoded_buf = encoded_buf[-200:]

            # --- Trigger VG-MLP mutation via φ_id drop ---
            if (pending_observed is None and step - last_observed_mut >= 30
                    and len(encoded_buf) >= 35
                    and step - last_phi_eval >= 15):
                last_phi_eval = step
                recent = encoded_buf[-35:]
                # Pad shapes (observer dim may have changed)
                min_dim = min(z.shape[0] for z in recent)
                coords = np.stack([z[:min_dim] for z in recent], axis=0).astype(np.float32)
                if coords.shape[1] > 3:
                    centered = coords - coords.mean(0)
                    u, s, vt = np.linalg.svd(centered, full_matrices=False)
                    coords3 = (u[:, :3] * s[:3]).astype(np.float32)
                else:
                    coords3 = coords
                try:
                    r = client.emergence(coords3.tolist())
                    phi = float(r.get('phi_id_approx', 0.0))
                    phi_history.append(phi)
                    if phi_ema is None:
                        phi_ema = phi
                        phi_baseline = phi
                    else:
                        phi_ema = 0.4 * phi + 0.6 * phi_ema
                        if phi_ema > phi_baseline:
                            phi_baseline = phi_ema
                        drop = phi_ema - phi_baseline
                        if drop < -0.01:
                            # Mutation observed
                            snap = copy.deepcopy(observed)
                            val_before = eval_acc(observed, va, max_batches=4)
                            target = bottleneck_layer(observed)
                            if observed.layers[target].out_features < 128:
                                observed.widen_layer(target, 12, opt_obs, use_gradmax=True)
                                encoded_buf.clear()  # shape change
                                pending_observed = {
                                    'step': step, 'snap': snap,
                                    'val_before': val_before,
                                    'verify_step': step + 30,
                                    'target': target,
                                }
                                observed_mutations.append({
                                    'step': step, 'type': 'try_widen',
                                    'target': target, 'phi_drop': drop,
                                    'topo_after': observed.topology(),
                                })
                                last_observed_mut = step
                                if verbose:
                                    print(f'[step {step:4d}] OBSERVED widen L{target} '
                                          f'phi_drop={drop:.3f} topo={observed.topology()}')
                except Exception as e:
                    pass

            # --- Verify observed mutation ---
            if pending_observed and step >= pending_observed['verify_step']:
                val_after = eval_acc(observed, va, max_batches=4)
                gain = val_after - pending_observed['val_before']
                if gain > 1e-3:
                    observed_mutations[-1]['status'] = 'accepted'
                    observed_mutations[-1]['gain'] = gain
                    if verbose:
                        print(f'[step {step:4d}] OBSERVED accept gain={gain:+.3f}')
                else:
                    # Rollback
                    snap = pending_observed['snap']
                    observed.layers = snap.layers
                    observed.head = snap.head
                    opt_obs = torch.optim.Adam(observed.parameters(), lr=1e-3)
                    observed_mutations[-1]['status'] = 'rejected'
                    observed_mutations[-1]['gain'] = gain
                    encoded_buf.clear()
                pending_observed = None

            # --- Observer try grow via SSVG plateau ---
            if step - last_observer_grow_try >= 50:
                mut = observer.try_grow(
                    opt_observer, observer_ssvg_buf,
                    plateau_window=30, plateau_eps=5e-5,
                    grow_k=4, max_channels=32,
                )
                last_observer_grow_try = step
                if mut is not None:
                    mut['step'] = step
                    observer_mutations.append(mut)
                    encoded_buf.clear()  # observer dim change
                    if verbose:
                        print(f'[step {step:4d}] OBSERVER widen conv{mut["target_conv"]} '
                              f'topo {mut["old_topology"]} → {mut["new_topology"]}')

            prev_img = curr_img
            step += 1

    final_acc = eval_acc(observed, va, max_batches=20)

    # Correlation timing : compute steps of mutations
    observed_steps = [m['step'] for m in observed_mutations]
    observer_steps = [m['step'] for m in observer_mutations]

    # Phase lock test : pour chaque observed mut, distance min à observer mut
    if observed_steps and observer_steps:
        min_dists = [min(abs(o - obs) for obs in observer_steps) for o in observed_steps]
        avg_min_dist = float(np.mean(min_dists))
    else:
        avg_min_dist = None

    return {
        'seed': seed,
        'final_acc': final_acc,
        'observed_topology_final': observed.topology(),
        'observed_params_final': observed.num_params(),
        'observer_topology_final': observer.topology(),
        'observer_params_final': observer.num_params(),
        'n_observed_mutations': len(observed_mutations),
        'n_observed_accepted': sum(1 for m in observed_mutations if m.get('status') == 'accepted'),
        'n_observer_mutations': len(observer_mutations),
        'observed_mut_steps': observed_steps,
        'observer_mut_steps': observer_steps,
        'phi_history': phi_history,
        'avg_min_mut_distance': avg_min_dist,
        'observed_mutations': observed_mutations,
        'observer_mutations': observer_mutations,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint5d')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 5D — co-évolution observer × observed ===')
    print(f'Seeds: {seeds}')

    client = DynamoscopeClient()
    print(f'Dynamoscope: {client.health()["device"]}')

    rows = []
    for seed in seeds:
        print(f'\n[seed {seed}] ...')
        t0 = time.time()
        r = run_coevolution(seed, client, verbose=True)
        elapsed = time.time() - t0
        r['elapsed'] = elapsed
        rows.append(r)
        print(f'\n  acc={r["final_acc"]:.3f}  ({elapsed:.1f}s)')
        print(f'  OBSERVED  : {r["n_observed_mutations"]} muts ({r["n_observed_accepted"]} acc) · '
              f'topo {r["observed_topology_final"]} · params {r["observed_params_final"]}')
        print(f'  OBSERVER  : {r["n_observer_mutations"]} muts · '
              f'topo {r["observer_topology_final"]} · params {r["observer_params_final"]}')
        print(f'  avg_min_mut_dist : {r["avg_min_mut_distance"]}')
        print(f'  observed steps: {r["observed_mut_steps"]}')
        print(f'  observer steps: {r["observer_mut_steps"]}')

    # Aggregate
    md = ['# Sprint 5D — Co-évolution observer × observed\n']
    md.append('## Summary\n')
    md.append('| seed | acc | observed muts | observer muts | obsv topo | observer topo | phase lock dist |')
    md.append('|---|---|---|---|---|---|---|')
    for r in rows:
        dist = f'{r["avg_min_mut_distance"]:.1f}' if r['avg_min_mut_distance'] is not None else '—'
        md.append(
            f'| {r["seed"]} | {r["final_acc"]:.3f} | '
            f'{r["n_observed_mutations"]}({r["n_observed_accepted"]}acc) | '
            f'{r["n_observer_mutations"]} | '
            f'{r["observed_topology_final"]} | '
            f'{r["observer_topology_final"]} | '
            f'{dist} |'
        )

    accs = [r['final_acc'] for r in rows]
    obsd_muts = [r['n_observed_mutations'] for r in rows]
    obsr_muts = [r['n_observer_mutations'] for r in rows]
    md.append(f'\n**Mean acc** : {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    md.append(f'**Mean observed muts** : {np.mean(obsd_muts):.1f}')
    md.append(f'**Mean observer muts** : {np.mean(obsr_muts):.1f}')

    # Phase lock analysis
    dists = [r['avg_min_mut_distance'] for r in rows if r['avg_min_mut_distance'] is not None]
    if dists:
        md.append(f'**Avg phase-lock distance** : {np.mean(dists):.1f} steps')
        md.append('(plus petite = mutations corrélées dans le temps = co-évolution couplée)')

    (args.out / 'coevolution_results.json').write_text(
        json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'coevolution_report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
