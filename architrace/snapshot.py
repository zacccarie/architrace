"""
Snapshot architecture VG → vecteur features fixed-size.
Sortie : trajectoire ND injectable dans dynamoscope.

Sprint 4 — option meta-loop : architecture VG = système dynamique.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn


MAX_LAYERS = 8           # padding fixe pour vecteur uniforme
PER_LAYER_FEATS = 6      # width, mean|w|, spectral_radius, eff_rank, mean|grad|, mean|bias|
GLOBAL_FEATS = 10
TOTAL_DIM = MAX_LAYERS * PER_LAYER_FEATS + GLOBAL_FEATS  # 58


@dataclass
class ArchSnapshot:
    """Représentation sérialisable d'un état architecture VG à un instant t."""
    step: int
    wall_time: float
    train_loss: float
    val_loss: float
    total_params: int
    depth: int
    mutations_total: int
    accepted: int
    rejected: int
    frozen: int
    distill_remaining: int
    layer_widths: list[int] = field(default_factory=list)
    layer_mean_abs_w: list[float] = field(default_factory=list)
    layer_spectral_radius: list[float] = field(default_factory=list)
    layer_effective_rank: list[float] = field(default_factory=list)
    layer_mean_abs_grad: list[float] = field(default_factory=list)
    layer_mean_abs_bias: list[float] = field(default_factory=list)

    def to_vec(self) -> list[float]:
        """Flat vecteur padded à TOTAL_DIM. Layers manquants → zéros.
        NaN/Inf coerced to 0.0 (UMAP/sklearn refusent valeurs non-finies)."""
        def _safe(x: float) -> float:
            x = float(x)
            return 0.0 if (math.isnan(x) or math.isinf(x)) else x

        vec: list[float] = []
        n = min(len(self.layer_widths), MAX_LAYERS)
        for i in range(MAX_LAYERS):
            if i < n:
                vec.extend([
                    _safe(self.layer_widths[i]),
                    _safe(self.layer_mean_abs_w[i]),
                    _safe(self.layer_spectral_radius[i]),
                    _safe(self.layer_effective_rank[i]),
                    _safe(self.layer_mean_abs_grad[i]),
                    _safe(self.layer_mean_abs_bias[i]),
                ])
            else:
                vec.extend([0.0] * PER_LAYER_FEATS)
        vec.extend([
            _safe(self.total_params),
            _safe(self.depth),
            _safe(self.val_loss),
            _safe(self.train_loss),
            _safe(self.step),
            _safe(self.mutations_total),
            _safe(self.accepted),
            _safe(self.rejected),
            _safe(self.frozen),
            _safe(self.distill_remaining),
        ])
        assert len(vec) == TOTAL_DIM, f"vec len mismatch: {len(vec)} != {TOTAL_DIM}"
        return vec


@torch.no_grad()
def _effective_rank(weight: torch.Tensor) -> float:
    """Rang effectif = exp(entropie normalisée des valeurs singulières).
    Compris dans [1, min(m,n)]. Mesure dispersion spectrale."""
    if weight.numel() == 0 or min(weight.shape) == 0:
        return 0.0
    sv = torch.linalg.svdvals(weight.float())
    total = sv.sum().item()
    if total < 1e-12:
        return 0.0
    p = (sv / total).clamp_min(1e-12)
    entropy = -(p * p.log()).sum().item()
    return math.exp(entropy)


@torch.no_grad()
def _spectral_radius(weight: torch.Tensor) -> float:
    """Plus grande valeur singulière (≈ Lipschitz constant)."""
    if weight.numel() == 0 or min(weight.shape) == 0:
        return 0.0
    sv = torch.linalg.svdvals(weight.float())
    return sv.max().item() if sv.numel() > 0 else 0.0


def snapshot(
    model: nn.Module,
    *,
    step: int,
    train_loss: float,
    val_loss: float,
    mutations: list[dict],
    frozen: bool,
    distill_remaining: int,
    include_head: bool = True,
) -> ArchSnapshot:
    """Extrait snapshot complet de l'état architecture + dynamique.

    model: VerifyGrowNet (expose .layers ModuleList + .head Linear)
    mutations: liste mutations cumulées (status ∈ {pending, accepted, rejected, ...})
    """
    layers = list(model.layers)
    if include_head and hasattr(model, "head"):
        layers = layers + [model.head]

    widths: list[int] = []
    mean_abs_w: list[float] = []
    spectral: list[float] = []
    eff_rank: list[float] = []
    mean_abs_grad: list[float] = []
    mean_abs_bias: list[float] = []

    for layer in layers:
        if not isinstance(layer, nn.Linear):
            continue
        w = layer.weight.data
        widths.append(layer.out_features)
        mean_abs_w.append(w.abs().mean().item())
        spectral.append(_spectral_radius(w))
        eff_rank.append(_effective_rank(w))
        g = layer.weight.grad
        mean_abs_grad.append(g.abs().mean().item() if g is not None else 0.0)
        b = layer.bias.data if layer.bias is not None else None
        mean_abs_bias.append(b.abs().mean().item() if b is not None else 0.0)

    accepted = sum(1 for m in mutations if m.get("status") == "accepted")
    rejected = sum(1 for m in mutations if m.get("status") == "rejected")

    return ArchSnapshot(
        step=step,
        wall_time=time.time(),
        train_loss=float(train_loss),
        val_loss=float(val_loss),
        total_params=int(sum(p.numel() for p in model.parameters())),
        depth=len(widths),
        mutations_total=len(mutations),
        accepted=accepted,
        rejected=rejected,
        frozen=int(frozen),
        distill_remaining=int(distill_remaining),
        layer_widths=widths,
        layer_mean_abs_w=mean_abs_w,
        layer_spectral_radius=spectral,
        layer_effective_rank=eff_rank,
        layer_mean_abs_grad=mean_abs_grad,
        layer_mean_abs_bias=mean_abs_bias,
    )


def var_names() -> list[str]:
    """Noms des TOTAL_DIM features pour SINDy / inspect dynamoscope."""
    names: list[str] = []
    for i in range(MAX_LAYERS):
        names.extend([
            f"L{i}_width",
            f"L{i}_mean|w|",
            f"L{i}_spectral",
            f"L{i}_eff_rank",
            f"L{i}_mean|grad|",
            f"L{i}_mean|bias|",
        ])
    names.extend([
        "total_params", "depth", "val_loss", "train_loss",
        "step", "mut_total", "accepted", "rejected",
        "frozen", "distill_remaining",
    ])
    return names


def snapshot_to_dict(s: ArchSnapshot) -> dict:
    """Sérialisation JSON-safe (pour WebSocket + fichier trace)."""
    return asdict(s)
