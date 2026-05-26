"""
Génère les figures de synthèse pour les sprints architrace.

Esthétique : fond noir, couleurs émissives (cf. instructions globales).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Esthétique globale
plt.rcParams.update({
    'figure.facecolor': '#0a0a0a',
    'axes.facecolor': '#0a0a0a',
    'axes.edgecolor': '#888',
    'axes.labelcolor': '#ddd',
    'xtick.color': '#ddd',
    'ytick.color': '#ddd',
    'text.color': '#ddd',
    'axes.titlecolor': '#fff',
    'grid.color': '#222',
    'grid.linestyle': ':',
    'font.size': 10,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'legend.framealpha': 0.0,
    'legend.edgecolor': '#888',
})

OUT = Path(__file__).parent
COLORS = {
    'cyan':    '#00d4ff',
    'magenta': '#ff00aa',
    'green':   '#00ff88',
    'orange':  '#ff8800',
    'yellow':  '#ffdd00',
    'red':     '#ff3344',
    'purple':  '#aa66ff',
    'white':   '#ffffff',
    'gray':    '#888888',
}


def fig1_llm_scaling():
    """Figure 1 : MSE Δpp vs log(params) sur 4 modèles."""
    params = np.array([0.41, 1.4, 2.8, 6.9])  # B params
    mse_delta = np.array([2.34, 1.07, 0.80, 0.43])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(params, mse_delta, marker='o', markersize=10,
                color=COLORS['cyan'], linewidth=2, label='MSE two-halves recipe')

    # Annotate each point
    for p, d in zip(params, mse_delta):
        ax.annotate(f'  Δ={d:+.2f}', (p, d), color=COLORS['white'],
                     fontsize=10, ha='left', va='center')

    # Linear fit log-linéaire
    log_p = np.log10(params)
    slope, intercept = np.polyfit(log_p, mse_delta, 1)
    x_fit = np.logspace(-0.4, 1.7, 50)
    y_fit = slope * np.log10(x_fit) + intercept
    ax.plot(x_fit, y_fit, '--', color=COLORS['orange'], alpha=0.6,
            label=f'fit: Δpp = {slope:.2f}·log₁₀(B) + {intercept:.2f}')

    # Crossover marker
    crossover_B = 10 ** (-intercept / slope)
    ax.axhline(0, color=COLORS['gray'], linestyle=':', alpha=0.5)
    ax.axvline(crossover_B, color=COLORS['red'], linestyle=':', alpha=0.5,
                label=f'Δ=0 crossover ≈ {crossover_B:.1f}B')

    ax.set_xlabel('Paramètres modèle (milliards)')
    ax.set_ylabel('Δ perplexité vs baseline α=0')
    ax.set_title('Sprint 19/20 — MSE two-halves recipe : dégradation décroît log-linéairement')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 50)
    ax.set_ylim(-0.5, 3)

    plt.tight_layout()
    plt.savefig(OUT / 'fig1_llm_scaling.png', dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print('  ✓ fig1_llm_scaling.png')


def fig2_infonce_u_shape():
    """Figure 2 : Sweet spot α pour InfoNCE per-token (Sprint 20C)."""
    alphas = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    pp_410m = np.array([27.480, 27.354, 27.342, 27.488, 27.482, 27.967])
    std_410m = np.array([0.974, 0.904, 0.822, 0.879, 0.853, 0.723])
    delta_410m = pp_410m - pp_410m[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(alphas, delta_410m, yerr=std_410m, marker='o', markersize=10,
                color=COLORS['magenta'], linewidth=2, capsize=5,
                label='pythia-410M (3 seeds × 6 α)')
    ax.axhline(0, color=COLORS['gray'], linestyle=':', alpha=0.5)

    # Highlight sweet spot
    best_idx = np.argmin(delta_410m)
    ax.scatter([alphas[best_idx]], [delta_410m[best_idx]], s=200,
                facecolors='none', edgecolors=COLORS['green'],
                linewidths=2.5, label=f'sweet spot α={alphas[best_idx]} (Δ={delta_410m[best_idx]:+.2f})')

    ax.set_xlabel('α — poids aux loss')
    ax.set_ylabel('Δ perplexité vs α=0')
    ax.set_title('Sprint 20C — U-shape : aux loss InfoNCE bénéfique seulement à faible α')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / 'fig2_infonce_u_shape.png', dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print('  ✓ fig2_infonce_u_shape.png')


def fig3_substrate_matrix():
    """Figure 3 : Heatmap substrate × task × verdict."""
    substrates = ['MLP\n(S11.x)', 'CNN/Mixer\n(S18C)', 'ViT\n(S18C)',
                  'LLM-text\n(S20C)', 'LLM-text+warmup\n(S20D)',
                  'LLM-text+world-LoRA\n(S20F)', 'LLM-world-task\n(S20G)']
    # Score : positif = recipe aide, négatif = nuit, 0 = neutre
    scores = np.array([[+1.0], [+0.6], [-0.3], [+0.05], [+0.3],
                        [-0.4], [-0.8]])
    labels = ['103× speedup', '8-12× speedup', '0.7× inverse',
              'Δpp = -0.14', 'Δpp = -0.61\n(smooth-prior)',
              'Δpp = +0.53\n(forgetting)', 'Δmse = +78%\n(anti-aligned)']

    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(scores, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

    ax.set_yticks(range(len(substrates)))
    ax.set_yticklabels(substrates)
    ax.set_xticks([])

    for i, (lbl, score) in enumerate(zip(labels, scores.flatten())):
        ax.text(0, i, lbl, ha='center', va='center',
                color='black' if abs(score) > 0.3 else 'white',
                fontsize=10, fontweight='bold')

    ax.set_title('Recipe self-coherence : verdict par substrat × tâche',
                 fontsize=12, pad=15)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05,
                        shrink=0.6)
    cbar.set_label('← nuit                  neutre                  aide →',
                   color='#ddd')
    cbar.ax.xaxis.set_tick_params(color='#ddd')

    plt.tight_layout()
    plt.savefig(OUT / 'fig3_substrate_matrix.png', dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print('  ✓ fig3_substrate_matrix.png')


def fig4_world_warmup_vanishing():
    """Figure 4 : Smooth-prior bonus disparaît à scale (S20DE 410M vs 1.4B)."""
    conditions = ['none\n(α=0)', 'fresh\n(α=0.1)', 'world\n(Lorenz)', 'gaussian\n(walk)']
    delta_410m = [0, -0.138, -0.607, -0.505]
    delta_1_4b = [0, -0.107, +0.016, +0.117]
    std_410m = [0.97, 0.82, 1.01, 1.03]
    std_1_4b = [0.98, 0.96, 0.92, 0.94]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, delta_410m, width, yerr=std_410m,
                    color=COLORS['cyan'], label='pythia-410M', capsize=4,
                    edgecolor=COLORS['white'], linewidth=0.5)
    bars2 = ax.bar(x + width/2, delta_1_4b, width, yerr=std_1_4b,
                    color=COLORS['magenta'], label='pythia-1.4B', capsize=4,
                    edgecolor=COLORS['white'], linewidth=0.5)

    ax.axhline(0, color=COLORS['gray'], linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Δ perplexité vs baseline')
    ax.set_title('Sprint 20D/E — Smooth-prior warmup aide à 410M, disparaît à 1.4B+')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate value labels
    for bar, val in zip(bars1, delta_410m):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val >= 0 else -0.15),
                f'{val:+.2f}', ha='center', color=COLORS['cyan'], fontsize=9)
    for bar, val in zip(bars2, delta_1_4b):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val >= 0 else -0.15),
                f'{val:+.2f}', ha='center', color=COLORS['magenta'], fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / 'fig4_warmup_vanishing.png', dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print('  ✓ fig4_warmup_vanishing.png')


def fig5_world_task_destruction():
    """Figure 5 : Sprint 20G — aux loss DÉGRADE world task (anti-aligned)."""
    alphas = np.array([0.0, 0.1, 1.0])
    val_mse = np.array([0.0203, 0.0362, 0.1585])
    std_mse = np.array([0.0074, 0.019, 0.056])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(['α=0.0\n(baseline)', 'α=0.1\n(S20C sweet spot)',
                    'α=1.0\n(MSE-recipe weight)'],
                   val_mse, yerr=std_mse, capsize=8,
                   color=[COLORS['green'], COLORS['yellow'], COLORS['red']],
                   edgecolor=COLORS['white'], linewidth=0.5)

    for bar, val, pct in zip(bars, val_mse, [0, 78, 700]):
        h = bar.get_height()
        label = f'{val:.4f}' + (f'\n(+{pct}%)' if pct > 0 else '')
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                label, ha='center', color=COLORS['white'], fontsize=10)

    ax.set_ylabel('val MSE (next Lorenz coord)')
    ax.set_title('Sprint 20G — LLM-on-world-task : aux loss DÉGRADE (contradiction structurelle)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 0.25)

    plt.tight_layout()
    plt.savefig(OUT / 'fig5_world_task_destruction.png', dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print('  ✓ fig5_world_task_destruction.png')


def fig6_sprint_timeline():
    """Figure 6 : Timeline des 25 sprints avec verdicts."""
    sprints = [
        (1, 'snapshot+bridge', 'infra', 'green'),
        (2, 'reverse trigger', 'finding', 'green'),
        (3, 'adaptive encoder', 'finding', 'green'),
        (4, '1/f² spectral', 'finding', 'green'),
        (5, 'strange loop', 'partial', 'yellow'),
        (6, 'transfer FashionMNIST', 'finding', 'green'),
        (7, 'self-coherence cost', 'finding', 'green'),
        (8, 'hierarchical', 'partial', 'yellow'),
        (9, 'self-Lyapunov', 'finding', 'green'),
        (10, 'mutable observer', 'partial', 'yellow'),
        (11, 'EXAPTATION 103×', 'major', 'green'),
        (12, 'cross-arch ViT', 'partial', 'yellow'),
        (13, 'circuit distributed', 'partial', 'yellow'),
        (14, 'LLM recipe WIP', 'fail', 'orange'),
        (15, 'drift-norm 74×', 'major', 'green'),
        (16, 'CIFAR-100 + GPT2', 'mixed', 'yellow'),
        (17, 'TinyGPT pretrain', 'fail', 'orange'),
        (18, 'per-token + cross-arch', 'partial', 'yellow'),
        (19, 'LLM scale 7B', 'finding', 'green'),
        ('20A', '6.9B scaling', 'finding', 'green'),
        ('20B', 'full fine-tune', 'partial', 'yellow'),
        ('20C', 'InfoNCE recipe', 'rescue', 'green'),
        ('20D', 'world warmup', 'partial', 'yellow'),
        ('20E', 'gaussian control', 'reframe', 'yellow'),
        ('20F', 'whole-model warmup', 'fail', 'red'),
        ('20G', 'LLM-on-world', 'fail', 'red'),
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    color_map = {'green': COLORS['green'], 'yellow': COLORS['yellow'],
                  'orange': COLORS['orange'], 'red': COLORS['red'],
                  'major': COLORS['cyan'], 'rescue': COLORS['cyan']}

    for i, (sprint, title, kind, color) in enumerate(sprints):
        col = color_map.get(color, COLORS['gray'])
        ax.barh(i, 1, height=0.6, color=col, edgecolor=COLORS['white'], linewidth=0.3)
        ax.text(1.1, i, f'S{sprint} — {title}', va='center',
                color=COLORS['white'], fontsize=9)
        ax.text(0.5, i, kind.upper(), ha='center', va='center',
                color='black' if color != 'red' else 'white',
                fontsize=8, fontweight='bold')

    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, len(sprints) - 0.5)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Trajectoire des sprints architrace — 25 expériences', pad=15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['green'], label='Finding solide'),
        Patch(facecolor=COLORS['yellow'], label='Partiel / nuancé'),
        Patch(facecolor=COLORS['orange'], label='Échec faible'),
        Patch(facecolor=COLORS['red'], label='Échec net'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', ncol=2)

    plt.tight_layout()
    plt.savefig(OUT / 'fig6_sprint_timeline.png', dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print('  ✓ fig6_sprint_timeline.png')


def fig7_infonce_compare_mse():
    """Figure 7 : MSE vs InfoNCE comparison sur scaling."""
    params = [0.41, 1.4]
    mse_delta = [2.34, 1.07]
    infonce_delta = [-0.138, -0.107]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(params))
    width = 0.35

    bars1 = ax.bar(x - width/2, mse_delta, width, color=COLORS['red'],
                    label='MSE two-halves (Sprint 19)',
                    edgecolor=COLORS['white'], linewidth=0.5)
    bars2 = ax.bar(x + width/2, infonce_delta, width, color=COLORS['green'],
                    label='InfoNCE per-token α=0.1 (Sprint 20C)',
                    edgecolor=COLORS['white'], linewidth=0.5)

    ax.axhline(0, color=COLORS['gray'], linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}B' for p in params])
    ax.set_xlabel('Paramètres modèle')
    ax.set_ylabel('Δ perplexité vs baseline α=0')
    ax.set_title('Sprint 19 vs 20C — Signal alignment flip le signe')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars1, mse_delta):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                f'+{val:.2f}', ha='center', color=COLORS['red'], fontsize=10)
    for bar, val in zip(bars2, infonce_delta):
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.15,
                f'{val:+.2f}', ha='center', color=COLORS['green'], fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT / 'fig7_mse_vs_infonce.png', dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print('  ✓ fig7_mse_vs_infonce.png')


def fig8_exaptation_speedup_comparison():
    """Figure 8 : Speedup init surp par substrat."""
    substrates = ['MLP\nMNIST\nS11.3', 'MLP\ndrift-norm\nS15C', 'CNN\nCIFAR-100\nS16B',
                  'ResNet\nMixer\nS18C', 'ViT\nCIFAR\nS18C',
                  'LLM-410M\npredictor-only\nS20D', 'LLM-410M\nwhole-model\nS20F']
    speedups = [103, 74, 8.1, 10, 0.7, 1.02, 0.99]
    colors_bar = [COLORS['green'], COLORS['green'], COLORS['green'],
                  COLORS['green'], COLORS['red'], COLORS['orange'], COLORS['orange']]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(substrates, speedups, color=colors_bar,
                   edgecolor=COLORS['white'], linewidth=0.5, log=True)

    ax.axhline(1, color=COLORS['gray'], linestyle=':', alpha=0.7,
                label='1× (no speedup)')
    ax.set_ylabel('Init surprise speedup (world / fresh)\nlog-scale')
    ax.set_title('Exaptation S11.x : speedup par substrat — universalité partielle')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, which='both')
    ax.set_ylim(0.5, 200)

    for bar, val in zip(bars, speedups):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.15,
                f'{val}×', ha='center', color=COLORS['white'], fontsize=10,
                fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT / 'fig8_exaptation_speedup.png', dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print('  ✓ fig8_exaptation_speedup.png')


if __name__ == '__main__':
    print('Generating figures...')
    fig1_llm_scaling()
    fig2_infonce_u_shape()
    fig3_substrate_matrix()
    fig4_world_warmup_vanishing()
    fig5_world_task_destruction()
    fig6_sprint_timeline()
    fig7_infonce_compare_mse()
    fig8_exaptation_speedup_comparison()
    print(f'\n→ {OUT}/*.png')
