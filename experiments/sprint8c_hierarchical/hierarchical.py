"""
Sprint 8C — observers hiérarchiques (méta-méta-méta-observation).

Setup :
- Niveau 0 : VG-MLP observed (training MNIST)
- Niveau 1 : Observer1 (VG-CNN) observe render(VG-MLP) → encoded vector
- Niveau 2 : Observer2 (VG-CNN) observe render(Observer1) → encoded vector
- Optionnel Niveau 3 : Observer3 observe render(Observer2)

Chaque observer entraîne SSVG sur sa séquence d'images générées.

Question : higher-order observers learn anything ?
- Si SSVG loss observers convergent : oui, hiérarchie utile
- Si SSVG loss diverge ou stagne : récursion collapse
- Mesure : info inter-niveaux (correlation entre trajectoires encoded)
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import architrace  # noqa
from architrace.arch_to_image import render_arch_state
from architrace.bridge import DynamoscopeClient
from architrace.mutable_observer import MutableVGCNNObserver
from verifygrow_nn import VerifyGrowNet  # type: ignore


def load_mnist(n_train=4000, n_val=1000):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def run_hierarchy(n_levels: int, seed: int, client: DynamoscopeClient) -> dict:
    """Empile n_levels observers, chaque observe niveau précédent."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    # Observed VG-MLP
    observed = VerifyGrowNet(in_dim=784, out_dim=10, init_hidden=20, init_depth=2)
    opt_obs = torch.optim.Adam(observed.parameters(), lr=1e-3)

    # Hierarchy of observers
    observers = []
    opts_observer = []
    for lvl in range(n_levels):
        obs = MutableVGCNNObserver(channels=(4, 8), input_size=64)
        observers.append(obs)
        opts_observer.append(torch.optim.AdamW(obs.parameters(), lr=1e-3))

    # Buffers : pour chaque niveau, séquence images observées
    # Niveau 0 voit observed render. Niveau k voit render(observer[k-1])
    ssvg_losses_per_level: list[list[float]] = [[] for _ in range(n_levels)]
    encoded_traces_per_level: list[list[np.ndarray]] = [[] for _ in range(n_levels)]
    prev_imgs_per_level: list[np.ndarray | None] = [None] * n_levels

    step = 0
    for ep in range(5):
        for x, y in tr:
            # Observed train
            observed.train()
            opt_obs.zero_grad()
            logits = observed(x)
            task_loss = F.cross_entropy(logits, y)
            task_loss.backward()
            opt_obs.step()

            # Niveau 0 : render observed
            curr_img = render_arch_state(observed, image_size=64)

            for lvl in range(n_levels):
                # SSVG step si on a prev_img à ce niveau
                if prev_imgs_per_level[lvl] is not None:
                    ssvg_loss = observers[lvl].train_ssvg_step(
                        prev_imgs_per_level[lvl], curr_img, opts_observer[lvl]
                    )
                    ssvg_losses_per_level[lvl].append(ssvg_loss)

                # Encode pour trace
                if step % 5 == 0:
                    z = observers[lvl].encode(curr_img[None, ...])[0]
                    encoded_traces_per_level[lvl].append(z)

                prev_imgs_per_level[lvl] = curr_img

                # Next level : render de cet observer-ci
                if lvl < n_levels - 1:
                    curr_img = render_arch_state(observers[lvl], image_size=64)

            step += 1

    # Eval task acc final
    observed.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in va:
            pred = observed(x).argmax(-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    final_acc = correct / max(1, total)

    # Analyse : pour chaque niveau, ssvg loss trajectory + finale + variance
    level_stats = []
    for lvl in range(n_levels):
        ll = ssvg_losses_per_level[lvl]
        if not ll:
            level_stats.append(None)
            continue
        level_stats.append({
            'level': lvl,
            'n_ssvg_steps': len(ll),
            'ssvg_loss_init': float(np.mean(ll[:10])) if len(ll) >= 10 else ll[0],
            'ssvg_loss_final': float(np.mean(ll[-10:])) if len(ll) >= 10 else ll[-1],
            'ssvg_loss_min': float(min(ll)),
            'ssvg_loss_std_late': float(np.std(ll[-30:])) if len(ll) >= 30 else 0,
            'n_encoded': len(encoded_traces_per_level[lvl]),
            'encoded_dim': int(encoded_traces_per_level[lvl][0].shape[0]) if encoded_traces_per_level[lvl] else 0,
        })

    # Inter-level correlation : entre traces encodées des différents niveaux
    inter_level_corrs = []
    if n_levels >= 2:
        for i in range(n_levels - 1):
            t_i = encoded_traces_per_level[i]
            t_j = encoded_traces_per_level[i + 1]
            if t_i and t_j:
                n = min(len(t_i), len(t_j))
                # Use mean over batch as scalar trajectory per level
                traj_i = np.array([z.mean() for z in t_i[:n]])
                traj_j = np.array([z.mean() for z in t_j[:n]])
                if traj_i.std() > 1e-9 and traj_j.std() > 1e-9:
                    corr = float(np.corrcoef(traj_i, traj_j)[0, 1])
                else:
                    corr = 0.0
                inter_level_corrs.append({'i': i, 'j': i+1, 'pearson': corr})

    return {
        'seed': seed,
        'n_levels': n_levels,
        'final_observed_acc': final_acc,
        'level_stats': level_stats,
        'inter_level_corrs': inter_level_corrs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', type=str, default='1,2,3')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint8c')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    levels = [int(l.strip()) for l in args.levels.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 8C — hierarchical observers ===')
    print(f'Levels: {levels}  Seeds: {seeds}')

    client = DynamoscopeClient()
    print(f'Dynamoscope: {client.health()["device"]}')

    rows = []
    for n_lvl in levels:
        print(f'\n--- n_levels = {n_lvl} ---')
        for seed in seeds:
            print(f'[seed {seed}] ...', end='', flush=True)
            t0 = time.time()
            r = run_hierarchy(n_lvl, seed, client)
            r['elapsed'] = time.time() - t0
            rows.append(r)
            ls_str = ' | '.join(
                f'L{ls["level"]} init={ls["ssvg_loss_init"]:.4f}→{ls["ssvg_loss_final"]:.4f}'
                for ls in r['level_stats'] if ls is not None
            )
            print(f'  acc={r["final_observed_acc"]:.3f}  ({r["elapsed"]:.1f}s)')
            for ls in r['level_stats']:
                if ls:
                    print(f'    L{ls["level"]}: ssvg {ls["ssvg_loss_init"]:.4f} → '
                          f'{ls["ssvg_loss_final"]:.4f}  (min {ls["ssvg_loss_min"]:.4f}, '
                          f'std_late {ls["ssvg_loss_std_late"]:.4f})')
            if r['inter_level_corrs']:
                corrs = ' | '.join(f'L{c["i"]}↔L{c["j"]}={c["pearson"]:.3f}'
                                   for c in r['inter_level_corrs'])
                print(f'    inter-level corr: {corrs}')

    # Render report
    md = ['# Sprint 8C — observers hiérarchiques\n']
    md.append('## Per-run\n')
    md.append('| n_lvls | seed | acc | levels SSVG | inter-level corr |')
    md.append('|---|---|---|---|---|')
    for r in rows:
        lvls_str = ' / '.join(
            f'L{ls["level"]}:{ls["ssvg_loss_final"]:.4f}'
            for ls in r['level_stats'] if ls is not None
        )
        corr_str = ' / '.join(
            f'{c["i"]}↔{c["j"]}={c["pearson"]:.2f}'
            for c in r['inter_level_corrs']
        ) or '—'
        md.append(
            f'| {r["n_levels"]} | {r["seed"]} | {r["final_observed_acc"]:.3f} | '
            f'{lvls_str} | {corr_str} |'
        )

    # Aggregate by n_levels
    md.append('\n## Aggregate par niveau profondeur\n')
    md.append('| level | mean SSVG final | std SSVG final | converge ? |')
    md.append('|---|---|---|---|')
    by_lvl_idx = {}
    for r in rows:
        for ls in r['level_stats']:
            if ls is not None:
                by_lvl_idx.setdefault(ls['level'], []).append(ls)
    for lvl in sorted(by_lvl_idx):
        runs = by_lvl_idx[lvl]
        finals = [r['ssvg_loss_final'] for r in runs]
        inits = [r['ssvg_loss_init'] for r in runs]
        delta = np.mean(finals) - np.mean(inits)
        conv = 'YES' if delta < -0.01 * np.mean(inits) else 'NO (stagnant)'
        md.append(
            f'| L{lvl} | {np.mean(finals):.4f}±{np.std(finals):.4f} | '
            f'{np.mean(finals)-np.mean(inits):+.4f} init delta | {conv} |'
        )

    md.append('\n## Verdict\n')
    # Question : higher-order learn ?
    if 0 in by_lvl_idx and 1 in by_lvl_idx:
        l0_final = np.mean([r['ssvg_loss_final'] for r in by_lvl_idx[0]])
        l1_final = np.mean([r['ssvg_loss_final'] for r in by_lvl_idx[1]])
        l0_init = np.mean([r['ssvg_loss_init'] for r in by_lvl_idx[0]])
        l1_init = np.mean([r['ssvg_loss_init'] for r in by_lvl_idx[1]])
        md.append(f'- L0 SSVG : {l0_init:.4f} → {l0_final:.4f}')
        md.append(f'- L1 SSVG : {l1_init:.4f} → {l1_final:.4f}')
        l0_drop = (l0_init - l0_final) / max(l0_init, 1e-6) * 100
        l1_drop = (l1_init - l1_final) / max(l1_init, 1e-6) * 100
        md.append(f'- L0 % drop : {l0_drop:.1f}%  ·  L1 % drop : {l1_drop:.1f}%')
        if l1_drop > 10:
            md.append('- **L1 apprend** : higher-order observer captures structure utile')
        else:
            md.append('- **L1 stagne** : récursion saturée, méta-méta-info marginal')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
