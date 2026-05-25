# Sprint 5C — Strange loop self-référence

## Verdict global

- N runs: 3
- **Final val_acc** : 0.877 ± 0.005
- **Final self_surprise** : 2.5397 ± 0.6690
- **Final state_drift** : 0.001 ± 0.001
- **Final state_norm** : 0.779 ± 0.067

## Per-run

| seed | acc | params | task L→F | surprise L→F | drift L→F | norm L→F |
|---|---|---|---|---|---|---|
| 0 | 0.875 | 27690 | 2.31→0.35 | 0.042→2.783 | 0.04→0.00 | 0.80→0.80 |
| 1 | 0.883 | 27690 | 2.29→0.28 | 0.045→3.209 | 0.01→0.00 | 0.85→0.85 |
| 2 | 0.873 | 27690 | 2.33→0.18 | 0.046→1.626 | 0.04→0.00 | 0.69→0.69 |

## Interprétation dynamique

- **Self-surprise trend mixte** : pas de tendance claire inter-seed
- **Tâche apprise malgré boucle étrange** : acc moyenne 0.877. Auto-référence n'empêche pas convergence task.