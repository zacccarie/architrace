"""
Sprint 20F — LLM entier entraîné comme world model (S18C-equivalent).

Gap S20D : seul predictor head voyait Lorenz. LLM weights non touchés par
le world signal. S18C cross-arch vision = TOUT le modèle voit world.

Design :
  Phase 1 World : bypass token embedding, inject Lorenz coords via
                  coord_proj. LoRA fine-tune attention layers + coord_head.
                  Train next-coord regression.
  Phase 2 Self  : restore standard token embedding, train text + InfoNCE
                  per-token α=0.1.

LoRA weights carry over phase 1 → phase 2 = signal world embedded in
attention adapters. Vraie exaptation cross-substrate.

Conditions :
  none        : α=0 text-only baseline
  fresh       : α=0.1 text + InfoNCE, no warmup (= S20C fresh)
  world_full  : phase 1 LLM+LoRA on Lorenz, phase 2 text + InfoNCE
                (NEW vs S20D world which only warmed predictor head)
"""
from __future__ import annotations
import argparse
import json
import sys as _sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import architrace  # noqa

_sys.path.insert(0, str(architrace.DYNAMOSCOPE_PATH))
from backend.systems import lorenz  # type: ignore


def prepare_text_data(tokenizer, seq_len=128, n_train=400, n_val=80, seed=0):
    from sklearn.datasets import fetch_20newsgroups
    print('Loading 20 newsgroups subset...')
    data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    text = ' '.join(data.data[:80])
    enc = tokenizer(text, return_tensors='pt', truncation=False)
    ids = enc['input_ids'][0]
    print(f'Corpus : {len(ids)} tokens')
    rng = np.random.RandomState(seed)
    starts_train = rng.randint(0, len(ids) - seq_len - 1, size=n_train)
    starts_val = rng.randint(0, len(ids) - seq_len - 1, size=n_val)
    train = torch.stack([ids[s:s + seq_len] for s in starts_train])
    val = torch.stack([ids[s:s + seq_len] for s in starts_val])
    return train, val


def make_lorenz_seqs(seq_len=128, n_seqs=400, seed=0):
    lz = lorenz(n=3000, dt=0.01)
    coords = np.array(lz['raw_coords'], dtype=np.float32)
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    rng = np.random.RandomState(seed)
    starts = rng.randint(0, len(coords) - seq_len - 1, size=n_seqs)
    seqs = np.stack([coords[s:s + seq_len] for s in starts])
    return torch.from_numpy(seqs)  # (n_seqs, seq_len, 3)


def info_nce_loss(pred, target, temperature=0.1):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.t() / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)


def phase_world_llm(model, coord_proj, coord_head, lorenz_seqs, device,
                     steps=100, batch=4):
    """LLM + LoRA trained to predict next Lorenz coord. Bypass token embed."""
    print(f'[phase 1 world LLM] {steps} steps on Lorenz, batch={batch}')
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(coord_proj.parameters()) + list(coord_head.parameters())
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    for step in range(steps):
        idx = torch.randint(0, len(lorenz_seqs), (batch,))
        seqs = lorenz_seqs[idx].to(device)  # (B, T, 3)
        opt.zero_grad()
        emb_fp32 = coord_proj(seqs)  # (B, T, d_model) fp32
        emb = emb_fp32.to(dtype=torch.float16)
        out = model(inputs_embeds=emb, output_hidden_states=True)
        hidden = out.hidden_states[-1].float()
        pred = coord_head(hidden)  # (B, T, 3)
        loss = ((pred[:, :-1] - seqs[:, 1:]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if (step + 1) % 25 == 0:
            print(f'    [world] step {step+1}/{steps}  mse={loss.item():.4f}')


def run(model_name: str, condition: str, seed: int, device,
        max_steps: int = 80, weight: float = 0.1,
        temperature: float = 0.1, state_dim: int = 64,
        world_steps: int = 100) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f'\n[{condition}/seed={seed}] Loading {model_name} LoRA...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    cfg = base.config
    target_modules = ['query_key_value', 'dense'] if 'pythia' in model_name.lower() \
                     else ['c_attn', 'c_proj']
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.1, bias='none', task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora_cfg)
    model.to(device)

    d_model = cfg.hidden_size

    # Phase 1 World (only condition='world_full')
    if condition == 'world_full':
        coord_proj_p1 = nn.Linear(3, d_model).to(device, dtype=torch.float32)
        coord_head_p1 = nn.Linear(d_model, 3).to(device, dtype=torch.float32)
        lorenz_seqs = make_lorenz_seqs(seq_len=128, n_seqs=400, seed=seed)
        phase_world_llm(model, coord_proj_p1, coord_head_p1, lorenz_seqs,
                         device, steps=world_steps, batch=4)
        # discard coord_proj/head — keep LoRA weights warmed

    # Phase 2 modules
    state_proj = nn.Linear(d_model, state_dim).to(device, dtype=torch.float32)
    predictor = nn.Sequential(
        nn.Linear(state_dim, state_dim * 2), nn.GELU(),
        nn.Linear(state_dim * 2, state_dim),
    ).to(device, dtype=torch.float32)

    active_weight = 0.0 if condition == 'none' else weight

    train_seqs, val_seqs = prepare_text_data(tokenizer, seq_len=128, n_train=400,
                                              n_val=80, seed=seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(state_proj.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)

    BATCH = 4
    history = {'lm_loss': [], 'surp': []}
    t0 = time.time()
    step = 0
    while step < max_steps:
        perm = torch.randperm(len(train_seqs))
        for b in range(0, len(perm), BATCH):
            if step >= max_steps:
                break
            idx = perm[b:b + BATCH]
            batch = train_seqs[idx].to(device)
            x = batch[:, :-1]; y = batch[:, 1:]
            opt.zero_grad()
            out = model(input_ids=x, output_hidden_states=True)
            logits = out.logits
            hidden = out.hidden_states[-1].float()
            lm_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       y.reshape(-1))
            states = state_proj(hidden)
            pred_states = predictor(states)
            target = states[:, 1:].detach()
            pred = pred_states[:, :-1]
            pred_flat = pred.reshape(-1, state_dim)
            target_flat = target.reshape(-1, state_dim)
            surp = info_nce_loss(pred_flat, target_flat, temperature)
            loss = lm_loss + active_weight * surp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            history['lm_loss'].append(lm_loss.item())
            history['surp'].append(surp.item())
            step += 1
            if step % 20 == 0:
                print(f'    [phase 2] step {step}/{max_steps}  '
                      f'lm={lm_loss.item():.3f}  surp={surp.item():.3f}')
    train_time = time.time() - t0

    model.eval()
    val_loss_sum = 0; n = 0
    with torch.no_grad():
        for b in range(0, len(val_seqs), BATCH):
            batch = val_seqs[b:b + BATCH].to(device)
            x = batch[:, :-1]; y = batch[:, 1:]
            out = model(input_ids=x)
            val_loss_sum += F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                y.reshape(-1)
            ).item()
            n += 1
    val_loss = val_loss_sum / max(1, n)
    val_pp = float(np.exp(val_loss))

    return {
        'model_name': model_name,
        'condition': condition,
        'weight_used': active_weight,
        'seed': seed,
        'world_steps': world_steps if condition == 'world_full' else 0,
        'val_perplexity': val_pp,
        'final_lm_loss': float(np.mean(history['lm_loss'][-10:])),
        'final_surp': float(np.mean(history['surp'][-10:])),
        'init_surp': float(np.mean(history['surp'][:10])),
        'train_time': train_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='EleutherAI/pythia-410m')
    ap.add_argument('--conditions', type=str, default='none,fresh,world_full')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--max-steps', type=int, default=80)
    ap.add_argument('--world-steps', type=int, default=100)
    ap.add_argument('--weight', type=float, default=0.1)
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint20f')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conds = [c.strip() for c in args.conditions.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'=== Sprint 20F — LLM whole-model world warmup ===')
    print(f'Model: {args.model}  Conds: {conds}  Seeds: {seeds}  '
          f'Steps: {args.max_steps}  WorldSteps: {args.world_steps}\n')

    rows = []
    for cond in conds:
        print(f'\n--- {cond} ---')
        for seed in seeds:
            r = run(args.model, cond, seed, device, args.max_steps,
                    args.weight, world_steps=args.world_steps)
            rows.append(r)
            print(f'  ✓ {cond}/seed={seed}  val_pp={r["val_perplexity"]:.3f}  '
                  f'surp {r["init_surp"]:.3f}→{r["final_surp"]:.3f}  '
                  f'({r["train_time"]:.0f}s)')

    by_c = {}
    for r in rows:
        by_c.setdefault(r['condition'], []).append(r)

    md = ['# Sprint 20F — LLM whole-model world warmup\n']
    md.append(f'Model : `{args.model}`\n')
    md.append(f'α : {args.weight}  World steps : {args.world_steps}\n')
    md.append('## Aggregate par condition\n')
    md.append('| condition | val_pp | surp init→final | Δ pp vs none |')
    md.append('|---|---|---|---|')
    none_pp = float(np.mean([r['val_perplexity'] for r in by_c.get('none', [])])) \
        if 'none' in by_c else 0
    for cond in conds:
        if cond not in by_c:
            continue
        runs = by_c[cond]
        pps = [r['val_perplexity'] for r in runs]
        i_sps = [r['init_surp'] for r in runs]
        f_sps = [r['final_surp'] for r in runs]
        delta = (np.mean(pps) - none_pp) if none_pp else 0
        md.append(
            f'| {cond} | {np.mean(pps):.3f}±{np.std(pps):.3f} | '
            f'{np.mean(i_sps):.3f}→{np.mean(f_sps):.3f} | '
            f'{delta:+.3f} |'
        )

    md.append('\n## Verdict whole-model world warmup\n')
    if 'world_full' in by_c and 'fresh' in by_c:
        wf_pp = float(np.mean([r['val_perplexity'] for r in by_c['world_full']]))
        fr_pp = float(np.mean([r['val_perplexity'] for r in by_c['fresh']]))
        wf_init = float(np.mean([r['init_surp'] for r in by_c['world_full']]))
        fr_init = float(np.mean([r['init_surp'] for r in by_c['fresh']]))
        speedup = fr_init / max(wf_init, 1e-9)
        md.append(f'- world_full init surp : {wf_init:.3f}')
        md.append(f'- fresh init surp : {fr_init:.3f}')
        md.append(f'- **Speedup init surp : {speedup:.2f}×**')
        md.append(f'- world_full val_pp : {wf_pp:.3f}')
        md.append(f'- fresh val_pp : {fr_pp:.3f}')
        md.append(f'- **Δ pp (world_full vs fresh) : {wf_pp - fr_pp:+.3f}**')
        md.append('')
        if speedup > 1.5 and (wf_pp - fr_pp) < -0.2:
            md.append('→ **EXAPTATION VRAIE CONFIRMÉE EN LLM** : whole-model warmup'
                       ' déclenche speedup + val_pp gain. Gap S20D résolu.')
        elif speedup > 1.5:
            md.append('→ Speedup init OK mais pas gain val_pp final.'
                       ' Predictor exaptable, mais bénéfice ne se transfère pas.')
        elif (wf_pp - fr_pp) < -0.3:
            md.append('→ Gain val_pp sans speedup init : smooth-prior à nouveau,'
                       ' mais via LoRA cette fois.')
        else:
            md.append('→ Pas de bénéfice net. LLM substrate vraiment'
                       ' incompatible avec exaptation S11.x.')

    md.append('\n## Comparaison S20D (predictor-only warmup)\n')
    md.append('| condition | S20D Δ pp (410M) | S20F Δ pp |')
    md.append('|---|---|---|')
    md.append(f'| none | 0.000 | {0.000:+.3f} |')
    if 'fresh' in by_c and none_pp:
        fr = float(np.mean([r["val_perplexity"] for r in by_c["fresh"]])) - none_pp
        md.append(f'| fresh | -0.138 | {fr:+.3f} |')
    if 'world_full' in by_c and none_pp:
        wf = float(np.mean([r["val_perplexity"] for r in by_c["world_full"]])) - none_pp
        md.append(f'| world (predictor-only S20D / whole-model S20F) | -0.607 | {wf:+.3f} |')

    (args.out / f'results_{args.model.replace("/", "_")}.json').write_text(
        json.dumps({'rows': rows, 'args': vars(args)}, default=str, indent=2)
    )
    (args.out / f'report_{args.model.replace("/", "_")}.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
