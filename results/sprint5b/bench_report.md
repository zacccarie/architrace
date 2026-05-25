# Sprint 5B — Visual emergence trigger

## Summary (mean ± std)

| trigger | acc | params | n_fires | first_fire | accept/reject |
|---|---|---|---|---|---|
| **loss_plateau** | 0.876±0.004 | 16330±0 | 0.0±0.0 | nan | 0.0/0.0 |
| **direct_emergence** | 0.902±0.002 | 48530±4554 | 3.3±0.5 | 152 | 3.0/0.0 |
| **visual_emergence** | 0.898±0.006 | 45310±0 | 3.0±0.0 | 144 | 3.0/0.0 |

## Per-run

| trigger | seed | acc | params | topo | fires | first | phi_range |
|---|---|---|---|---|---|---|---|
| loss_plateau | 0 | 0.875 | 16330 | [20, 20] | 0 | — | — |
| loss_plateau | 1 | 0.882 | 16330 | [20, 20] | 0 | — | — |
| loss_plateau | 2 | 0.872 | 16330 | [20, 20] | 0 | — | — |
| direct_emergence | 0 | 0.905 | 54970 | [68, 20] | 4 | 117 | [0.088, 0.647] |
| direct_emergence | 1 | 0.900 | 45310 | [56, 20] | 3 | 192 | [0.199, 0.788] |
| direct_emergence | 2 | 0.902 | 45310 | [56, 20] | 3 | 147 | [0.321, 0.899] |
| visual_emergence | 0 | 0.895 | 45310 | [56, 20] | 3 | 134 | [0.254, 0.996] |
| visual_emergence | 1 | 0.906 | 45310 | [56, 20] | 3 | 164 | [0.213, 0.774] |
| visual_emergence | 2 | 0.892 | 45310 | [56, 20] | 3 | 134 | [0.165, 0.782] |