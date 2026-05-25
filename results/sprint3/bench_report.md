# Sprint 3 — VG-CNN vs ResNet50 encoder bench

Trajectoire Lorenz rendered en frames 64×64 RGB, encodée → dynamoscope analyses.

| encoder | seed | params | latent_dim | enc_time | lyap | corr_dim | RQA_DET | sindy_r2 | phi_id | spectral_slope |
|---|---|---|---|---|---|---|---|---|---|---|
| vg_tiny | 0 | 498 | 8 | 0.24s | 0.082 | 1.378 | 0.805 | 0.057 | 0.477 | -1.559 |
| vg_small | 0 | 6,362 | 32 | 0.13s | 0.073 | 1.871 | 0.847 | 0.184 | 0.227 | -1.586 |
| vg_medium | 0 | 24,234 | 64 | 0.24s | 0.083 | 1.585 | 0.885 | 0.144 | 0.053 | -1.826 |
| resnet50 | 0 | 23,508,032 | 2048 | 1.27s | 0.067 | 2.551 | 0.884 | 0.124 | 0.000 | -1.837 |
| raw_lorenz | 0 | 0 | 3 | 0.00s | 0.036 | 1.131 | 1.000 | 0.242 | 0.000 | -2.663 |
| vg_tiny | 1 | 498 | 8 | 0.03s | 0.091 | 0.852 | 0.719 | 0.042 | 0.755 | -1.400 |
| vg_small | 1 | 6,362 | 32 | 0.02s | 0.099 | 1.329 | 0.878 | 0.014 | 0.000 | -1.872 |
| vg_medium | 1 | 24,234 | 64 | 0.03s | 0.078 | 1.316 | 0.916 | 0.261 | 0.000 | -1.804 |
| resnet50 | 1 | 23,508,032 | 2048 | 0.20s | 0.067 | 2.551 | 0.884 | 0.124 | 0.000 | -1.837 |
| raw_lorenz | 1 | 0 | 3 | 0.00s | 0.036 | 1.131 | 1.000 | 0.242 | 0.000 | -2.663 |

## Verdict

- **Ground truth (raw 3D Lorenz)** : Lyap=0.036  corr_dim=1.131  sindy_r2=0.242  spectral_slope=-2.663
- **vg_tiny** (498 params) — Lyap Δ=0.046, corr_dim Δ=0.247, sindy_r2=0.057
- **vg_small** (6,362 params) — Lyap Δ=0.038, corr_dim Δ=0.740, sindy_r2=0.184
- **vg_medium** (24,234 params) — Lyap Δ=0.048, corr_dim Δ=0.454, sindy_r2=0.144
- **resnet50** (23,508,032 params) — Lyap Δ=0.031, corr_dim Δ=1.420, sindy_r2=0.124