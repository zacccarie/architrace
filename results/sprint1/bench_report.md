# Sprint 1 — VG predictor vs LatentMLP fixe

## Per-system summary (mean ± std sur seeds)

| system | n_seeds | MLP-s pred | MLP-m pred | MLP-l pred | VG pred | VG params | MLP-m params | VG mutations |
|---|---|---|---|---|---|---|---|---|
| lorenz | 3 | 0.794±0.034 | 0.904±0.008 | 0.920±0.010 | 0.887±0.032 | 483±45 | 899±0 | 3.3±0.9 |
| rossler | 3 | 0.794±0.032 | 0.878±0.018 | 0.900±0.015 | 0.820±0.145 | 430±92 | 899±0 | 2.3±0.9 |
| van_der_pol | 3 | 0.300±0.056 | 0.556±0.022 | 0.675±0.013 | 0.557±0.271 | 515±157 | 899±0 | 3.3±0.9 |

## Per-run detail

| system | seed | MLP-s pred | MLP-m pred | MLP-l pred | VG pred | VG params | VG topo |
|---|---|---|---|---|---|---|---|
| lorenz | 0 | 0.751 (227p) | 0.908 (899p) | 0.934 (1795p) | 0.844 | 515 | [40, 8] |
| lorenz | 1 | 0.834 (227p) | 0.892 (899p) | 0.916 (1795p) | 0.919 | 515 | [40, 8] |
| lorenz | 2 | 0.798 (227p) | 0.911 (899p) | 0.909 (1795p) | 0.898 | 419 | [32, 8] |
| rossler | 0 | 0.772 (227p) | 0.899 (899p) | 0.893 (1795p) | 0.917 | 419 | [32, 8] |
| rossler | 1 | 0.839 (227p) | 0.880 (899p) | 0.921 (1795p) | 0.927 | 547 | [24, 16] |
| rossler | 2 | 0.770 (227p) | 0.855 (899p) | 0.888 (1795p) | 0.614 | 323 | [8, 24] |
| van_der_pol | 0 | 0.354 (227p) | 0.525 (899p) | 0.657 (1795p) | 0.826 | 707 | [16, 32] |
| van_der_pol | 1 | 0.323 (227p) | 0.565 (899p) | 0.687 (1795p) | 0.661 | 515 | [40, 8] |
| van_der_pol | 2 | 0.222 (227p) | 0.577 (899p) | 0.681 (1795p) | 0.186 | 323 | [8, 24] |

## Verdict per system

- **lorenz** : pred VG=0.887 vs MLP-m=0.904 → **MLP-medium** wins. VG uses 0.54× params of MLP-medium. (MLP-s=0.794, MLP-l=0.920)
- **rossler** : pred VG=0.820 vs MLP-m=0.878 → **MLP-medium** wins. VG uses 0.48× params of MLP-medium. (MLP-s=0.794, MLP-l=0.900)
- **van_der_pol** : pred VG=0.557 vs MLP-m=0.556 → **VG** wins. VG uses 0.57× params of MLP-medium. (MLP-s=0.300, MLP-l=0.675)