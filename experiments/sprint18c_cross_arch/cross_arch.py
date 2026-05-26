"""
Sprint 18C — S11.x exaptation cross-architecture vision.

3 archs : ResNet, ViT, MLP-Mixer. CIFAR-10. Test world→self speedup.
Drift contrôlé via noise (méthode S15C).

Si exaptation universel cross-arch → finding RÉELLEMENT universel.
"""
from __future__ import annotations
import argparse
import json
import sys as _sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import architrace  # noqa
from architrace.multi_arch_agents import ResNetSelfAgent, ViTSelfAgent, MixerSelfAgent
from architrace.encoder_vg import VGCNNEncoder

_sys.path.insert(0, str(architrace.DYNAMOSCOPE_PATH))
from backend.systems import lorenz  # type: ignore


def load_cifar10(n_train=4000, n_val=1500):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    tr = datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    va = datasets.CIFAR10('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def build_model(arch: str, device):
    if arch == 'resnet':
        return ResNetSelfAgent(in_ch=3, num_classes=10, state_dim=16, hidden=64,
                                observer_channels=(4, 8)).to(device)
    if arch == 'vit':
        return ViTSelfAgent(in_ch=3, num_classes=10, image_size=32, patch_size=4,
                             d_model=64, n_heads=4, n_layers=3, state_dim=16, hidden=64,
                             observer_channels=(4, 8)).to(device)
    if arch == 'mixer':
        return MixerSelfAgent(in_ch=3, num_classes=10, image_size=32, patch_size=4,
                               d_model=64, n_layers=3, state_dim=16, hidden=64,
                               observer_channels=(4, 8)).to(device)
    raise ValueError(arch)


def phase_world(model, opt, epochs, device):
    lz = lorenz(n=2000, dt=0.01)
    coords = np.array(lz['raw_coords'], dtype=np.float32)
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    state_dim = model.state_dim
    padded = np.zeros((len(coords), state_dim), dtype=np.float32)
    padded[:, :3] = coords
    padded_t = torch.from_numpy(padded).to(device)
    zero_img = torch.zeros(1, 3, 32, 32, device=device)
    for ep in range(epochs):
        perm = np.random.permutation(len(padded) - 1)
        for batch_start in range(0, len(perm), 64):
            batch_idx = perm[batch_start:batch_start + 64]
            for i in batch_idx:
                opt.zero_grad()
                _, pred = model(zero_img, padded_t[i])
                loss = ((pred.mean(0)[:3] - padded_t[i + 1][:3]) ** 2).mean()
                loss.backward()
                opt.step()


def phase_self(model, train_loader, val_loader, opt, weight, epochs, device, inject_noise):
    prev_state = model.self_state()
    history = {'surp_smoothed': []}
    buf = deque(maxlen=20)
    rng = torch.Generator(device=device).manual_seed(42)
    for ep in range(epochs):
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            current_raw = model.self_state()
            noise = torch.randn(current_raw.shape, device=device, generator=rng) * inject_noise
            current_noisy = current_raw + noise
            opt.zero_grad()
            logits, pred = model(x, prev_state)
            task = F.cross_entropy(logits, y)
            pred_mean = pred.mean(0)
            if pred_mean.shape[0] != current_noisy.shape[0]:
                mm = min(pred_mean.shape[0], current_noisy.shape[0])
                surp = ((pred_mean[:mm] - current_noisy[:mm]) ** 2).mean()
            else:
                surp = ((pred_mean - current_noisy) ** 2).mean()
            (task + weight * surp).backward()
            opt.step()
            prev_state = current_noisy
            buf.append(surp.item())
            history['surp_smoothed'].append(float(np.mean(buf)))
    model.eval()
    c = 0; t = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = model(vx, prev_state)
            c += (logits.argmax(-1) == vy).sum().item()
            t += vy.shape[0]
    return {'final_acc': c / max(1, t), 'history': history}


def run(arch: str, condition: str, seed: int, inject_noise: float):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tr_ds, va_ds = load_cifar10()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = build_model(arch, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    PHASE1_EPOCHS = 3
    PHASE2_EPOCHS = 2

    t0 = time.time()
    if condition == 'none':
        prev_state = model.self_state()
        for ep in range(PHASE1_EPOCHS):
            for x, y in tr:
                x = x.to(device); y = y.to(device)
                opt.zero_grad()
                logits, _ = model(x, prev_state)
                F.cross_entropy(logits, y).backward()
                opt.step()
    elif condition == 'world':
        phase_world(model, opt, PHASE1_EPOCHS, device)
    elif condition == 'fresh_self':
        phase_self(model, tr, va, opt, weight=1.0, epochs=PHASE1_EPOCHS, device=device,
                   inject_noise=inject_noise)
    else:
        raise ValueError(condition)
    p1_time = time.time() - t0

    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.time()
    res = phase_self(model, tr, va, opt2, weight=1.0, epochs=PHASE2_EPOCHS, device=device,
                     inject_noise=inject_noise)
    p2_time = time.time() - t0

    sm = res['history']['surp_smoothed']
    return {
        'arch': arch,
        'condition': condition,
        'seed': seed,
        'final_acc': res['final_acc'],
        'surp_init': float(sm[0]) if sm else None,
        'surp_final': float(sm[-1]) if sm else None,
        'p1_time': p1_time,
        'p2_time': p2_time,
        'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archs', type=str, default='resnet,vit,mixer')
    ap.add_argument('--noise', type=float, default=0.05)
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint18c')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archs = [a.strip() for a in args.archs.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    conditions = ['none', 'world', 'fresh_self']

    print(f'=== Sprint 18C — S11.x cross-architecture vision ===')
    print(f'Archs: {archs}  Seeds: {seeds}  Noise: {args.noise}\n')

    rows = []
    for arch in archs:
        for cond in conditions:
            print(f'--- {arch} | {cond} ---')
            for seed in seeds:
                r = run(arch, cond, seed, args.noise)
                rows.append(r)
                print(f'  seed={seed}  acc={r["final_acc"]:.3f}  '
                      f'surp_init={r["surp_init"]:.5f}  '
                      f'({r["p1_time"]:.0f}+{r["p2_time"]:.0f}s)  '
                      f'params={r["params"]:,}')

    md = ['# Sprint 18C — S11.x cross-architecture vision\n']
    md.append('## Aggregate par (arch, condition)\n')
    md.append('| arch | condition | acc | surp_init | speedup vs none |')
    md.append('|---|---|---|---|---|')

    by_ac = {}
    for r in rows:
        by_ac.setdefault((r['arch'], r['condition']), []).append(r)

    for arch in archs:
        none_surp = float(np.mean([r['surp_init'] for r in by_ac.get((arch, 'none'), [])])) \
            if (arch, 'none') in by_ac else 0
        for cond in conditions:
            key = (arch, cond)
            if key not in by_ac:
                continue
            runs = by_ac[key]
            accs = [r['final_acc'] for r in runs]
            surps = [r['surp_init'] for r in runs]
            speedup = none_surp / max(np.mean(surps), 1e-9) if none_surp else 0
            md.append(
                f'| {arch} | {cond} | {np.mean(accs):.3f}±{np.std(accs):.3f} | '
                f'{np.mean(surps):.5f}±{np.std(surps):.5f} | {speedup:.1f}× |'
            )

    md.append('\n## Verdict cross-arch\n')
    for arch in archs:
        if (arch, 'none') in by_ac and (arch, 'world') in by_ac:
            n_surp = np.mean([r['surp_init'] for r in by_ac[(arch, 'none')]])
            w_surp = np.mean([r['surp_init'] for r in by_ac[(arch, 'world')]])
            ratio = n_surp / max(w_surp, 1e-9)
            md.append(f'- **{arch}** : world→self speedup = **{ratio:.1f}×**')

    md.append('\nRappel : S11.3 MNIST MLP = 103×, S15C drift-norm = 74×, S16B CNN CIFAR-100 = 8.1×')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
