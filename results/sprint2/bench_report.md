# Sprint 2 — Triggers VG bench

## Per-trigger summary (mean ± std sur seeds)

| trigger | seeds | final_acc | final_params | n_fires | first_fire_step | accepted | rejected |
|---|---|---|---|---|---|---|---|
| **loss_plateau** | 3 | 0.876±0.004 | 16330±0 | 0.0±0.0 | — | 0.0±0.0 | 0.0±0.0 |
| **surprise** | 3 | 0.876±0.004 | 16330±0 | 0.0±0.0 | — | 0.0±0.0 | 0.0±0.0 |
| **emergence** | 3 | 0.902±0.002 | 48530±4554 | 3.3±0.5 | 152±31 | 3.0±0.0 | 0.0±0.0 |
| **lyapunov** | 3 | 0.902±0.008 | 45310±7887 | 3.3±0.5 | 177±56 | 2.7±0.9 | 0.3±0.5 |
| **composite** | 3 | 0.902±0.002 | 48530±4554 | 3.3±0.5 | 152±31 | 3.0±0.0 | 0.0±0.0 |

## Per-run detail

| trigger | seed | acc | params | topo | fires | first | acc/rej | reasons (first 2) |
|---|---|---|---|---|---|---|---|---|
| loss_plateau | 0 | 0.875 | 16330 | [20, 20] | 0 | — | 0/0 |  |
| loss_plateau | 1 | 0.882 | 16330 | [20, 20] | 0 | — | 0/0 |  |
| loss_plateau | 2 | 0.872 | 16330 | [20, 20] | 0 | — | 0/0 |  |
| surprise | 0 | 0.875 | 16330 | [20, 20] | 0 | — | 0/0 |  |
| surprise | 1 | 0.882 | 16330 | [20, 20] | 0 | — | 0/0 |  |
| surprise | 2 | 0.872 | 16330 | [20, 20] | 0 | — | 0/0 |  |
| emergence | 0 | 0.905 | 54970 | [68, 20] | 4 | 117 | 3/0 | phi drop -0.052 (raw=0.183 ema; phi drop -0.122 (raw=0.088 ema |
| emergence | 1 | 0.900 | 45310 | [56, 20] | 3 | 192 | 3/0 | phi drop -0.017 (raw=0.603 ema; phi drop -0.147 (raw=0.278 ema |
| emergence | 2 | 0.902 | 45310 | [56, 20] | 3 | 147 | 3/0 | phi drop -0.024 (raw=0.402 ema; phi drop -0.022 (raw=0.460 ema |
| lyapunov | 0 | 0.914 | 54970 | [68, 20] | 4 | 102 | 4/0 | lyap=0.079 > 0.05; lyap=0.088 > 0.05 |
| lyapunov | 1 | 0.897 | 35650 | [44, 20] | 3 | 192 | 2/1 | lyap=0.072 > 0.05; lyap=0.080 > 0.05 |
| lyapunov | 2 | 0.896 | 45310 | [56, 20] | 3 | 237 | 2/0 | lyap=0.103 > 0.05; lyap=0.090 > 0.05 |
| composite | 0 | 0.905 | 54970 | [68, 20] | 4 | 117 | 3/0 | vote=1.00 >= 1.0 | fired: ['em; vote=1.70 >= 1.0 | fired: ['em |
| composite | 1 | 0.900 | 45310 | [56, 20] | 3 | 192 | 3/0 | vote=1.70 >= 1.0 | fired: ['em; vote=1.70 >= 1.0 | fired: ['em |
| composite | 2 | 0.902 | 45310 | [56, 20] | 3 | 147 | 3/0 | vote=1.00 >= 1.0 | fired: ['em; vote=1.70 >= 1.0 | fired: ['em |

## Verdict

- **Meilleur final_acc** : `lyapunov` (0.902)
- **Plus petit modèle** : `loss_plateau` (16330 params)