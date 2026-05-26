# Sprint 14A — Layer-by-layer lesion

Trunk depth=3. Freeze 1 layer Phase 2, mesure self-coherence impact.

## Aggregate

| lesion | acc | surp_init | surp_final | surp_mean | surp_max |
|---|---|---|---|---|---|
| **no_lesion** | 0.844±0.007 | 0.0109±0.0080 | 0.0130±0.0021 | 0.0151±0.0009 | 0.0208±0.0028 |
| **layer_0** | 0.814±0.010 | 0.0109±0.0080 | 0.0118±0.0013 | 0.0130±0.0011 | 0.0185±0.0040 |
| **layer_1** | 0.840±0.006 | 0.0109±0.0080 | 0.0115±0.0012 | 0.0132±0.0012 | 0.0186±0.0040 |
| **layer_2** | 0.835±0.006 | 0.0109±0.0080 | 0.0109±0.0009 | 0.0126±0.0008 | 0.0179±0.0040 |

## Verdict

- **no_lesion** surp_final : 0.0130

- **layer_0** surp_final : 0.0118 (ratio 0.91× no_lesion)
- **layer_1** surp_final : 0.0115 (ratio 0.89× no_lesion)
- **layer_2** surp_final : 0.0109 (ratio 0.84× no_lesion)

- **Layer le plus critique** : 0 (ratio 0.91×)
- **Layer le moins critique** : 2 (ratio 0.84×)
- **Spread max/min** : 1.08×

  → **DISTRIBUTION UNIFORME** : toutes layers équivalentes pour self-coherence