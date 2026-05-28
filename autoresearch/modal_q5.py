"""Modal app — Q5 architrace : LLM-on-world-task avec MSE aux loss (au lieu InfoNCE).

Sprint 20G original (InfoNCE) : Δmse = +78% (dégrade).
Q5 hypothèse : MSE state-next aux loss = MIEUX que InfoNCE (smooth, aligned task type).

Workflow :
1. modal run modal_q5.py::smoke              # smoke test 1 seed 100 steps (~$0.30)
2. modal run modal_q5.py::full_sweep         # 3 weights × 3 seeds (~$3-5)

Volume `architrace-q5-cache` persiste HF cache pythia-410m entre runs.
"""
from __future__ import annotations
import modal
from pathlib import Path

app = modal.App("architrace-q5")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.9.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "transformers>=4.45",
        "peft>=0.13",
        "numpy>=1.26",
        "accelerate>=0.30",
    )
)

cache_vol = modal.Volume.from_name("architrace-q5-cache", create_if_missing=True)
HF_CACHE = "/hfcache"


@app.function(
    image=image,
    volumes={HF_CACHE: cache_vol},
    timeout=1800,
    gpu="H100",
)
def run_q5_iter(weight: float = 1.0, seed: int = 0, aux_type: str = "mse",
                model_name: str = "EleutherAI/pythia-410m", max_steps: int = 100,
                state_dim: int = 64) -> dict:
    """1 iter Q5 cloud. aux_type ∈ {'mse', 'infonce', 'none'}.

    Métrique décisionnelle : val_mse (next-coord regression loss).
    """
    import os
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

    device = torch.device("cuda")

    # --- Lorenz coord generator (inline, no architrace backend dep) ---
    def lorenz_trajectory(n=4000, dt=0.01, sigma=10, rho=28, beta=8 / 3):
        xs = np.zeros((n, 3), dtype=np.float64)
        xs[0] = [1.0, 1.0, 1.0]
        for i in range(1, n):
            x, y, z = xs[i - 1]
            xs[i, 0] = x + dt * sigma * (y - x)
            xs[i, 1] = y + dt * (x * (rho - z) - y)
            xs[i, 2] = z + dt * (x * y - beta * z)
        return xs.astype(np.float32)

    def make_lorenz_seqs(seq_len=128, n_train=400, n_val=80, seed=0):
        coords = lorenz_trajectory()
        coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
        rng = np.random.RandomState(seed)
        starts_tr = rng.randint(0, len(coords) - seq_len - 1, size=n_train)
        starts_va = rng.randint(0, len(coords) - seq_len - 1, size=n_val)
        tr = np.stack([coords[s:s + seq_len] for s in starts_tr])
        va = np.stack([coords[s:s + seq_len] for s in starts_va])
        return torch.from_numpy(tr), torch.from_numpy(va)

    def info_nce_loss(pred, target, temperature=0.1):
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        logits = pred @ target.t() / temperature
        labels = torch.arange(pred.size(0), device=pred.device)
        return F.cross_entropy(logits, labels)

    # --- Setup ---
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"\n=== Q5 iter aux={aux_type} weight={weight} seed={seed} ===")
    print(f"Loading {model_name}...")

    base = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, cache_dir=HF_CACHE,
    )
    cfg = base.config
    target_modules = ['query_key_value', 'dense']
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.1, bias='none', task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora_cfg)
    model.to(device)

    d_model = cfg.hidden_size
    coord_proj = nn.Linear(3, d_model).to(device, dtype=torch.float32)
    coord_head = nn.Linear(d_model, 3).to(device, dtype=torch.float32)
    state_proj = nn.Linear(d_model, state_dim).to(device, dtype=torch.float32)
    predictor = nn.Sequential(
        nn.Linear(state_dim, state_dim * 2), nn.GELU(),
        nn.Linear(state_dim * 2, state_dim),
    ).to(device, dtype=torch.float32)

    tr_seqs, va_seqs = make_lorenz_seqs(seq_len=128, n_train=400, n_val=80, seed=seed)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(coord_proj.parameters()) + list(coord_head.parameters())
    trainable += list(state_proj.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)

    BATCH = 4
    history = {"task_loss": [], "aux": []}
    t0 = time.time()
    step = 0
    while step < max_steps:
        perm = torch.randperm(len(tr_seqs))
        for b in range(0, len(perm), BATCH):
            if step >= max_steps:
                break
            idx = perm[b:b + BATCH]
            seqs = tr_seqs[idx].to(device)
            opt.zero_grad()
            emb_fp32 = coord_proj(seqs)
            emb = emb_fp32.to(dtype=torch.float16)
            out = model(inputs_embeds=emb, output_hidden_states=True)
            hidden = out.hidden_states[-1].float()
            pred_coord = coord_head(hidden)
            task_loss = ((pred_coord[:, :-1] - seqs[:, 1:]) ** 2).mean()

            # Aux loss : MSE or InfoNCE on next-state prediction
            states = state_proj(hidden)
            pred_states = predictor(states)
            target = states[:, 1:].detach()
            pred = pred_states[:, :-1]
            pred_flat = pred.reshape(-1, state_dim)
            target_flat = target.reshape(-1, state_dim)

            if aux_type == "mse":
                aux = F.mse_loss(pred_flat, target_flat)
            elif aux_type == "infonce":
                aux = info_nce_loss(pred_flat, target_flat, temperature=0.1)
            elif aux_type == "none":
                aux = torch.tensor(0.0, device=device)
            else:
                raise ValueError(f"aux_type inconnu : {aux_type}")

            loss = task_loss + weight * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            history["task_loss"].append(task_loss.item())
            history["aux"].append(aux.item())
            step += 1
            if step % 25 == 0:
                print(f"  step {step}/{max_steps}  task_mse={task_loss.item():.4f}  aux={aux.item():.3f}")
    train_time = time.time() - t0

    # --- Validation ---
    model.eval()
    val_mse_sum = 0.0; n_batches = 0
    with torch.no_grad():
        for b in range(0, len(va_seqs), BATCH):
            seqs = va_seqs[b:b + BATCH].to(device)
            emb = coord_proj(seqs).to(dtype=torch.float16)
            out = model(inputs_embeds=emb, output_hidden_states=True)
            hidden = out.hidden_states[-1].float()
            pred_coord = coord_head(hidden)
            mse = ((pred_coord[:, :-1] - seqs[:, 1:]) ** 2).mean().item()
            val_mse_sum += mse
            n_batches += 1
    val_mse = val_mse_sum / max(1, n_batches)

    cache_vol.commit()

    result = {
        "aux_type": aux_type,
        "weight": weight,
        "seed": seed,
        "val_mse": val_mse,
        "init_task_loss": float(np.mean(history["task_loss"][:10])),
        "final_task_loss": float(np.mean(history["task_loss"][-10:])),
        "init_aux": float(np.mean(history["aux"][:10])),
        "final_aux": float(np.mean(history["aux"][-10:])),
        "train_seconds": round(train_time, 1),
        "max_steps": max_steps,
    }
    print(f"\n=== Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


# === Local entrypoints ===


@app.local_entrypoint()
def smoke():
    """Smoke test 1 seed 100 steps avec MSE aux. ~3-5min H100 = $0.25."""
    r = run_q5_iter.remote(weight=1.0, seed=0, aux_type="mse", max_steps=100)
    print(f"\n=== Smoke result ===")
    print(f"val_mse = {r['val_mse']}")
    print(f"train_seconds = {r['train_seconds']}")


@app.local_entrypoint()
def full_sweep():
    """Sweep 3 conditions × 3 seeds. ~$3-5."""
    conditions = [
        ("none", 0.0),       # baseline pure task
        ("infonce", 1.0),    # match S20G original
        ("mse", 1.0),        # Q5 hypothèse
    ]
    _run_sweep_internal(conditions, seeds)


@app.local_entrypoint()
def alpha_sweep_old():
    """Sweep MSE α ∈ [0.1, 0.5, 1.0, 2.0] × 4 seeds. ~$2-3 cloud H100.

    Trace MSE aux dose-response curve. Si MSE harmful augmente avec α → trend confirmé.
    Si MSE flat ou inverted-U → nuance Q5 finding.
    """
    conditions = [
        ("none", 0.0),       # baseline
        ("mse", 0.1),
        ("mse", 0.5),
        ("mse", 1.0),
        ("mse", 2.0),
    ]
    seeds = [0, 1, 2, 3]  # 4 seeds × 5 conditions = 20 runs
    _run_sweep_internal(conditions, seeds)


def _run_sweep_internal(conditions, seeds):
    all_results = []
    for aux_type, weight in conditions:
        print(f"\n========= Aux={aux_type} w={weight} =========")
        for s in seeds:
            r = run_q5_iter.remote(weight=weight, seed=s, aux_type=aux_type, max_steps=100)
            all_results.append(r)

    # Aggregate
    import statistics
    print(f"\n=== Q5 Aggregate ===")
    for aux_type, weight in conditions:
        runs = [r for r in all_results if r["aux_type"] == aux_type and r["weight"] == weight]
        mses = [r["val_mse"] for r in runs]
        mean = statistics.mean(mses)
        std = statistics.stdev(mses) if len(mses) > 1 else 0.0
        print(f"  {aux_type:<10} α={weight}  val_mse = {mean:.5f} ± {std:.5f}  (n={len(runs)})")

    # Compare MSE vs InfoNCE vs baseline
    print(f"\n=== Verdict Q5 ===")
    base = [r["val_mse"] for r in all_results if r["aux_type"] == "none"]
    infonce = [r["val_mse"] for r in all_results if r["aux_type"] == "infonce"]
    mse = [r["val_mse"] for r in all_results if r["aux_type"] == "mse"]
    if base and infonce and mse:
        base_mean = statistics.mean(base)
        d_infonce = statistics.mean(infonce) - base_mean
        d_mse = statistics.mean(mse) - base_mean
        print(f"  Δ vs base : InfoNCE {d_infonce:+.5f}  |  MSE {d_mse:+.5f}")
        if d_mse < d_infonce:
            print(f"  → Q5 confirmé : MSE aux loss MIEUX que InfoNCE")
        elif d_mse > d_infonce:
            print(f"  → Q5 réfuté : MSE pire que InfoNCE")
        else:
            print(f"  → Q5 indistinguishable")
