"""
architrace.visual_trigger — Sprint 5B : observation visuelle de l'architecture.

VisualEmergenceTrigger observe le réseau qui pousse via render image →
VG-CNN encode → dynamoscope φ_id. Boucle complète :

    VG-MLP state
        ↓ arch_to_image render (64×64×3)
        ↓ VG-CNN encode (latent_dim D)
        ↓ buffer trace
        ↓ dynamoscope.emergence (φ_id)
        ↓ drop detection → trigger mutation VG-MLP

Fusion Sprint 2 (trigger) + Sprint 3 (VG-CNN encoder) + Sprint 4 (architecture
trajectory analysis). Méta-observation : observateur visuel de l'observé
qui se modifie.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .arch_to_image import render_arch_state
from .encoder_vg import VGCNNEncoder
from .triggers import Trigger, TrainHistory


class VisualEmergenceTrigger(Trigger):
    """φ_id sur trajectoire visuelle de l'architecture.

    Différence vs EmergenceTrigger :
    - EmergenceTrigger lit activations latentes directes (mean over batch)
    - VisualEmergenceTrigger render arch state en image → encode VG-CNN
      → latent vector. Le réseau est OBSERVÉ comme stimulus visuel.

    Avantage : invariant à dimensionalité du réseau observé (image fixed-size),
    et capture **structure visuelle** des poids/grads en plus de leurs valeurs.
    """

    name = "visual_emergence"

    def __init__(
        self,
        observer_encoder: VGCNNEncoder,
        window: int = 35,
        phi_drop_threshold: float = -0.01,
        ema_alpha: float = 0.4,
        cooldown: int = 30,
        capture_every: int = 5,
        eval_every: int = 15,
        image_size: int = 64,
        client=None,
    ):
        super().__init__(cooldown=cooldown)
        self.observer = observer_encoder
        self.window = window
        self.phi_drop_threshold = phi_drop_threshold
        self.ema_alpha = ema_alpha
        self.capture_every = capture_every
        self.eval_every = eval_every
        self.image_size = image_size
        self.client = client

        # Buffer encoded snapshots
        self.encoded_buf: list[np.ndarray] = []
        self.phi_ema: Optional[float] = None
        self.phi_baseline: Optional[float] = None
        self.phi_history: list[float] = []
        self.last_capture_step = -1
        self.last_eval_step = -1

    def capture(self, model, step: int):
        """Render arch + encode + append to buffer. Appelé via callback explicit."""
        if step - self.last_capture_step < self.capture_every:
            return
        self.last_capture_step = step
        img = render_arch_state(model, image_size=self.image_size)
        # encode expects (N, H, W, 3)
        z = self.observer.encode(img[None, ...])[0]  # (D,)
        self.encoded_buf.append(z)
        # Window cap
        max_keep = max(200, self.window * 4)
        if len(self.encoded_buf) > max_keep:
            self.encoded_buf = self.encoded_buf[-max_keep:]

    def should_grow(self, history: TrainHistory) -> tuple[bool, str]:
        if not self._cooldown_ok(history):
            return False, "cooldown"
        if len(self.encoded_buf) < self.window:
            return False, f"visual buf {len(self.encoded_buf)}/{self.window}"
        if self.client is None:
            return False, "no_client"
        if history.step - self.last_eval_step < self.eval_every:
            return False, "throttle"
        self.last_eval_step = history.step

        recent = self.encoded_buf[-self.window:]
        coords = np.stack(recent, axis=0).astype(np.float32)
        # PCA→3D pour passer dynamoscope/api/emergence (coords NxD attendu, mais 3D plus stable)
        if coords.shape[1] > 3:
            centered = coords - coords.mean(0)
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            coords3 = (u[:, :3] * s[:3]).astype(np.float32)
        else:
            coords3 = coords

        try:
            r = self.client.emergence(coords3.tolist())
            phi = float(r.get("phi_id_approx", 0.0))
        except Exception as e:
            return False, f"client_err: {str(e)[:50]}"

        self.phi_history.append(phi)

        if self.phi_ema is None:
            self.phi_ema = phi
            self.phi_baseline = phi
            return False, f"visual_phi_init={phi:.3f}"

        self.phi_ema = self.ema_alpha * phi + (1 - self.ema_alpha) * self.phi_ema
        if self.phi_ema > self.phi_baseline:
            self.phi_baseline = self.phi_ema

        drop = self.phi_ema - self.phi_baseline
        if drop < self.phi_drop_threshold:
            return True, (f"visual_phi drop {drop:.3f} "
                          f"(raw={phi:.3f} ema={self.phi_ema:.3f} peak={self.phi_baseline:.3f})")
        return False, f"visual_phi raw={phi:.3f} ema={self.phi_ema:.3f}"

    def reset_buffer(self):
        """Appelé après mutation pour clean buffer (encoding stale)."""
        self.encoded_buf.clear()
