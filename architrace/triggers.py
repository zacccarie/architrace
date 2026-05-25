"""
architrace.triggers — Sprint 2 : trigger pluggables pour mutation VG.

Remplace plateau detection (slope sur loss) par signaux dynamoscope.

Question : VG voit seulement courbe loss. Dynamoscope voit structure latente
(φ_id émergence, Lyapunov local, surprise prédictive). Coupler les deux =
trigger plus riche.

Interface :
    class Trigger:
        def should_grow(self, history: TrainHistory) -> tuple[bool, str]

Triggers livrés :
- LossPlateauTrigger        baseline = comportement VG actuel (slope sur train loss)
- SurpriseTrigger           rolling rollout n-step error spike détection
- EmergenceTrigger          φ_id de dynamoscope sur trace latents
- LyapunovLocalTrigger      local Lyapunov exceeds threshold (chaos = need capacity)
- CompositeTrigger          weighted sum d'autres triggers
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class TrainHistory:
    """État training observable par triggers.

    Tous les triggers ont accès aux mêmes signaux ; ils diffèrent dans ce qu'ils REGARDENT.
    """
    step: int
    loss_buffer: deque                          # rolling train loss
    val_loss: float                             # val_loss actuel
    val_loss_history: list[float]               # tous val_loss passés
    rollout_err_buffer: Optional[deque] = None  # rolling rollout n-step error
    latents_buffer: Optional[list[np.ndarray]] = None  # snapshots latent activations
    step_last_mutation: int = -10000


class Trigger(ABC):
    """Trigger ABC. Implémentations retournent (should_grow, reason)."""

    name: str = "abstract"
    cooldown: int = 60

    def __init__(self, cooldown: int = 60):
        self.cooldown = cooldown

    @abstractmethod
    def should_grow(self, history: TrainHistory) -> tuple[bool, str]:
        ...

    def _cooldown_ok(self, history: TrainHistory) -> bool:
        return history.step - history.step_last_mutation >= self.cooldown


# =============================================================================
# 1. LossPlateauTrigger — baseline
# =============================================================================
class LossPlateauTrigger(Trigger):
    """Slope sur train loss buffer. Comportement VG actuel."""

    name = "loss_plateau"

    def __init__(self, window: int = 80, eps: float = 5e-3, cooldown: int = 60):
        super().__init__(cooldown=cooldown)
        self.window = window
        self.eps = eps

    def should_grow(self, history: TrainHistory) -> tuple[bool, str]:
        if not self._cooldown_ok(history):
            return False, "cooldown"
        buf = history.loss_buffer
        if len(buf) < self.window:
            return False, f"buf {len(buf)}/{self.window}"
        arr = torch.tensor(list(buf))
        half = self.window // 2
        slope = (arr[-half:].mean() - arr[:half].mean()).abs().item()
        if slope < self.eps:
            return True, f"slope={slope:.4f} < {self.eps}"
        return False, f"slope={slope:.4f}"


# =============================================================================
# 2. SurpriseTrigger — rollout n-step error spike
# =============================================================================
class SurpriseTrigger(Trigger):
    """Trigger sur rolling rollout error. Si erreur croît au lieu de descendre → besoin capacité.

    Différent de plateau loss : capture moment où model NE PEUT PAS converger en n-step,
    même si single-step loss baisse.
    """

    name = "surprise"

    def __init__(self, window: int = 5, growth_threshold: float = 0.05, cooldown: int = 60):
        super().__init__(cooldown=cooldown)
        self.window = window
        self.growth_threshold = growth_threshold

    def should_grow(self, history: TrainHistory) -> tuple[bool, str]:
        if not self._cooldown_ok(history):
            return False, "cooldown"
        buf = history.rollout_err_buffer
        if buf is None or len(buf) < self.window + 2:
            return False, f"rollout_buf {len(buf) if buf else 0}"
        recent = np.array(list(buf)[-self.window:])
        older = np.array(list(buf)[-2*self.window:-self.window])
        if older.size == 0:
            return False, "older_empty"
        # Croissance relative
        rel_growth = (recent.mean() - older.mean()) / max(older.mean(), 1e-9)
        if rel_growth > self.growth_threshold:
            return True, f"rollout_err growth +{rel_growth*100:.1f}%"
        return False, f"growth={rel_growth*100:+.1f}%"


# =============================================================================
# 3. EmergenceTrigger — φ_id dynamoscope drop
# =============================================================================
class EmergenceTrigger(Trigger):
    """Trigger sur effondrement φ_id (Hoel) ou émergence below threshold.

    Hypothèse : si φ_id chute → représentation perd capacité causale émergente → grandir.
    Use bridge dynamoscope pour calculer φ_id sur fenêtre récente de latents.
    """

    name = "emergence"

    def __init__(
        self, window: int = 40, phi_drop_threshold: float = -0.1,
        ema_alpha: float = 0.3, cooldown: int = 120,
        client=None,  # DynamoscopeClient
    ):
        super().__init__(cooldown=cooldown)
        self.window = window
        self.phi_drop_threshold = phi_drop_threshold  # négatif = chute par rapport baseline
        self.ema_alpha = ema_alpha
        self.client = client
        self.phi_baseline: Optional[float] = None
        self.phi_ema: Optional[float] = None
        self.last_eval_step = -1
        self.eval_every = 20  # éviter appel HTTP tous les steps

    def should_grow(self, history: TrainHistory) -> tuple[bool, str]:
        if not self._cooldown_ok(history):
            return False, "cooldown"
        if history.latents_buffer is None or len(history.latents_buffer) < self.window:
            return False, f"latents {len(history.latents_buffer) if history.latents_buffer else 0}/{self.window}"
        if self.client is None:
            return False, "no_client"
        # Throttle
        if history.step - self.last_eval_step < self.eval_every:
            return False, "throttle"
        self.last_eval_step = history.step

        # Construit trace latents window
        recent = history.latents_buffer[-self.window:]
        coords = np.stack(recent, axis=0)
        # Project 3D si dim > 3
        try:
            if coords.shape[1] > 3:
                # PCA simple via SVD (évite HTTP UMAP pour speed)
                centered = coords - coords.mean(0)
                u, s, vt = np.linalg.svd(centered, full_matrices=False)
                coords3 = (u[:, :3] * s[:3]).astype(np.float32)
            else:
                coords3 = coords.astype(np.float32)
            r = self.client.emergence(coords3.tolist())
            phi = float(r.get("phi_id_approx", 0.0))
        except Exception as e:
            return False, f"client_err: {str(e)[:50]}"

        # EMA pour smoothing
        if self.phi_ema is None:
            self.phi_ema = phi
            self.phi_baseline = phi
            self.phi_history = [phi]
            return False, f"phi_init={phi:.3f}"

        self.phi_ema = self.ema_alpha * phi + (1 - self.ema_alpha) * self.phi_ema
        if not hasattr(self, 'phi_history'):
            self.phi_history = []
        self.phi_history.append(phi)

        # Track baseline = max φ vu jusqu'ici
        if self.phi_ema > self.phi_baseline:
            self.phi_baseline = self.phi_ema

        drop = self.phi_ema - self.phi_baseline
        if drop < self.phi_drop_threshold:
            return True, f"phi drop {drop:.3f} (raw={phi:.3f} ema={self.phi_ema:.3f} peak={self.phi_baseline:.3f})"
        return False, f"phi raw={phi:.3f} ema={self.phi_ema:.3f} peak={self.phi_baseline:.3f}"


# =============================================================================
# 4. LyapunovLocalTrigger — sensibilité conditions initiales
# =============================================================================
class LyapunovLocalTrigger(Trigger):
    """Trigger si Lyapunov local sur recent latents window > threshold.

    Intuition : Lyapunov > 0 = trajectoire latente devient chaotique sur recent steps.
    Représentation "explose" → besoin plus de capacité pour la dompter.
    """

    name = "lyapunov_local"

    def __init__(
        self, window: int = 60, lyap_threshold: float = 0.5,
        cooldown: int = 120, client=None,
    ):
        super().__init__(cooldown=cooldown)
        self.window = window
        self.lyap_threshold = lyap_threshold
        self.client = client
        self.last_eval_step = -1
        self.eval_every = 30

    def should_grow(self, history: TrainHistory) -> tuple[bool, str]:
        if not self._cooldown_ok(history):
            return False, "cooldown"
        if history.latents_buffer is None or len(history.latents_buffer) < self.window:
            return False, f"latents {len(history.latents_buffer) if history.latents_buffer else 0}/{self.window}"
        if self.client is None:
            return False, "no_client"
        if history.step - self.last_eval_step < self.eval_every:
            return False, "throttle"
        self.last_eval_step = history.step

        recent = history.latents_buffer[-self.window:]
        coords = np.stack(recent, axis=0).astype(np.float32)
        try:
            r = self.client.dynamics(coords.tolist())
            lyap = float(r.get("lyapunov", 0.0))
        except Exception as e:
            return False, f"client_err: {str(e)[:50]}"

        if lyap > self.lyap_threshold:
            return True, f"lyap={lyap:.3f} > {self.lyap_threshold}"
        return False, f"lyap={lyap:.3f}"


# =============================================================================
# 5. CompositeTrigger — vote pondéré
# =============================================================================
@dataclass
class TriggerWeight:
    trigger: Trigger
    weight: float = 1.0


class CompositeTrigger(Trigger):
    """Vote pondéré : declenche si sum(weights * fires) > threshold.

    Permet combiner forces des triggers individuels. Ex :
    - loss_plateau (poids 0.5) : signal nécessaire
    - emergence (poids 1.0) : signal causal
    - lyapunov (poids 0.7) : signal stabilité
    """

    name = "composite"

    def __init__(self, triggers: list[TriggerWeight], threshold: float = 1.0, cooldown: int = 60):
        super().__init__(cooldown=cooldown)
        self.weighted_triggers = triggers
        self.threshold = threshold

    def should_grow(self, history: TrainHistory) -> tuple[bool, str]:
        if not self._cooldown_ok(history):
            return False, "cooldown"
        total = 0.0
        fired_names = []
        reasons = []
        for tw in self.weighted_triggers:
            fired, reason = tw.trigger.should_grow(history)
            if fired:
                total += tw.weight
                fired_names.append(tw.trigger.name)
            reasons.append(f"{tw.trigger.name}={reason[:30]}")
        if total >= self.threshold:
            return True, f"vote={total:.2f} >= {self.threshold} | fired: {fired_names}"
        return False, f"vote={total:.2f}/{self.threshold}"
