"""Presentation-only rendering of saved rolling results; never fits a model.

The historical producer and its recorded source hash remain unchanged. Report
generation uses this renderer so chart improvements do not rewrite provenance.
"""
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs/figures/rolling_backtest.png'
COLORS = {'baseline': '#23678d', 'no_age': '#bd7021',
          'no_velocity_short': '#619758', 'no_velocity_all': '#9259a2'}
LABELS = {'baseline': 'Full-feature baseline', 'no_age': 'Without age',
          'no_velocity_short': 'Without velocity 6h/24h', 'no_velocity_all': 'Without all velocity'}


def build_figure(result):
    metrics = pd.DataFrame(result['metrics'])
    fairness = pd.DataFrame(result['fairness'])
    differences = pd.DataFrame(result['differences'])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for variant in COLORS:
        selected = (metrics.variant == variant) & (metrics.split == 'test')
        capacity = metrics[selected & (metrics.policy == 'fixed_review_5pct')].sort_values('test_month')
        frozen = metrics[selected & (metrics.policy == 'frozen_threshold')].sort_values('test_month')
        groups = fairness[(fairness.variant == variant) & (fairness.policy == 'fixed_review_5pct')].sort_values('test_month')
        style = dict(color=COLORS[variant], linewidth=2 if variant == 'baseline' else 1.5, markersize=5)
        axes[0, 0].plot(capacity.test_month+1, 100*capacity.recall, 'o-', label=LABELS[variant], **style)
        axes[0, 1].plot(frozen.test_month+1, 100*frozen.fpr, 'o-', **style)
        axes[1, 0].plot(groups.test_month+1, 100*groups.fpr_gap, 'o-', **style)
        if variant == 'baseline':
            for row in capacity.itertuples():
                axes[0, 0].annotate(f'{100*row.recall:.2f}%', (row.test_month+1, 100*row.recall),
                    xytext=(0, 9), textcoords='offset points', ha='center', color=COLORS[variant], fontsize=9)
    for offset, variant in enumerate(list(COLORS)[1:]):
        rows = differences[differences.variant == variant].sort_values('test_month')
        y = 100*rows.recall_topk_diff
        axes[1, 1].errorbar(rows.test_month+1+(offset-1)*.09, y,
            yerr=np.vstack([y-100*rows.recall_topk_ci_low, 100*rows.recall_topk_ci_high-y]),
            fmt='o', capsize=3, color=COLORS[variant], label=LABELS[variant])
    axes[0, 0].set(title='Recall at a fixed 5% review rate', ylabel='Recall (%)',
                   ylim=(0, 100), yticks=[0, 20, 40, 60, 80, 100])
    fpr_max = 100*metrics[(metrics.split == 'test') & (metrics.policy == 'frozen_threshold')].fpr.max()
    axes[0, 1].set(title='FPR after transferring the validation threshold', ylabel='FPR (%)',
                   ylim=(0, max(6, math.ceil(fpr_max*1.1))))
    axes[0, 1].axhline(5, color='#999999', ls='--', lw=1)
    gaps = 100*fairness[fairness.policy == 'fixed_review_5pct'].fpr_gap
    axes[1, 0].set(title='Age FPR gap at the same 5% review rate',
        ylabel='Older minus younger (percentage points)',
        ylim=(min(0, math.floor(gaps.min()*1.1)), max(1, math.ceil(gaps.max()*1.1))))
    axes[1, 0].axhline(0, color='#999999', ls='--', lw=1)
    limit = max(1, math.ceil(100*np.abs(differences[['recall_topk_ci_low', 'recall_topk_ci_high']].to_numpy()).max()*1.1))
    axes[1, 1].set(title='Recall change vs baseline (paired 95% interval)',
                   ylabel='Change (percentage points)', ylim=(-limit, limit))
    axes[1, 1].axhline(0, color='#999999', ls='--', lw=1)
    for ax in axes.ravel():
        ax.set(xticks=[5, 6, 7, 8], xlabel='Test month (1-based)')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Retrospective rolling checks | overlapping training windows', fontsize=12)
    return fig, axes


def render(result=None):
    if result is None:
        result = json.loads((ROOT/'outputs/rolling_backtest.json').read_text(encoding='utf-8'))
    fig, _ = build_figure(result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160)
    plt.close(fig)
    return OUTPUT


if __name__ == '__main__':
    print(render())
