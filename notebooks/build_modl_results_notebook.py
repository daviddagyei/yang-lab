"""Build the executed-results notebook for the 4x MoDL experiment."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "modl_results.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        markdown(
            """
# MoDL 4× MRI reconstruction results

This notebook records the completed **50-epoch / 94,900-step** MoDL experiment. It uses
the checkpoint selected solely by validation PSNR and evaluates it once on the held-out
test split. The implementation follows the MoDL pattern: a shared CNN regularizer is
alternated with conjugate-gradient data-consistency solves.

References: [MoDL paper](https://arxiv.org/abs/1712.02862) ·
[official MoDL repository](https://github.com/hkaggarwal/modl)
"""
        ),
        code(
            """
from pathlib import Path
import json, re
import matplotlib.pyplot as plt
import numpy as np
from html import escape
from IPython.display import HTML, Image, display

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
MODL_ROOT = Path('/data/david_agyei/nyu-dataset/derived/modl_4x_fp32')
RDBM_ROOT = Path('/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16')
EVAL = MODL_ROOT / 'evaluation'

def read_json(path):
    return json.loads(Path(path).read_text())

def show_table(rows, columns, precision=4):
    def render(value):
        if isinstance(value, float):
            return f'{value:.{precision}f}'
        return escape(str(value))
    header = ''.join(f'<th style="text-align:left;padding:4px 10px">{escape(label)}</th>' for _, label in columns)
    body = ''.join('<tr>' + ''.join(
        f'<td style="text-align:left;padding:4px 10px">{render(row[key])}</td>' for key, _ in columns
    ) + '</tr>' for row in rows)
    display(HTML(f'<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'))

selection = read_json(EVAL / 'best_validation_checkpoint.json')
selected_step = selection['checkpoint_step']
test = read_json(EVAL / 'test' / f'step-{selected_step:07d}' / 'summary.json')
rdbm = read_json(RDBM_ROOT / 'evaluation/test/step-0094900/summary.json')
progression = read_json(ROOT / 'results/modl_evaluation/qualitative/progression_metadata.json')
print(f"Selected checkpoint: step {selected_step:,}")
"""
        ),
        markdown(
            """
## Experiment integrity

The split is volume-level, so slices from a subject volume cannot leak between train,
validation, and test. All methods below use the same supplied sensitivity maps and the
same fixed 4× undersampling setup.
"""
        ),
        code(
            """
integrity = [
    {'Split': 'Train', 'Volumes': 120, 'Slices': 1898, 'Use': 'optimization'},
    {'Split': 'Validation', 'Volumes': 14, 'Slices': 220, 'Use': 'checkpoint selection'},
    {'Split': 'Test', 'Volumes': test['volume_count'], 'Slices': test['slice_count'], 'Use': 'final reporting'},
]
show_table(integrity, [(key, key) for key in ['Split', 'Volumes', 'Slices', 'Use']], precision=0)
print('Training schedule: 5 warm-up epochs with 1 unroll, then 45 epochs with 10 unrolls.')
"""
        ),
        markdown("## Training trajectory"),
        code(
            """
pattern = re.compile(r'step=(\\d+).*?loss=([0-9.eE+-]+).*?lambda=([0-9.eE+-]+)')
records = []
for log_name in ['train_k1.log', 'train_k10.log']:
    for line in (MODL_ROOT / log_name).read_text().splitlines():
        match = pattern.search(line)
        if match:
            records.append((int(match[1]), float(match[2]), float(match[3])))
records = sorted(dict((step, (loss, lam)) for step, loss, lam in records).items())
steps = np.array([row[0] for row in records])
losses = np.array([row[1][0] for row in records])
lambdas = np.array([row[1][1] for row in records])
window = 500
loss_smooth = np.array([np.median(losses[max(0, i-window+1):i+1]) for i in range(len(losses))])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(steps, loss_smooth, color='tab:blue')
axes[0].axvline(9_490, color='gray', linestyle='--', label='10-unroll stage begins')
axes[0].set(xlabel='Training step', ylabel='Loss (500-step median)', title='Optimization loss')
axes[0].set_yscale('log'); axes[0].legend()
axes[1].plot(steps, lambdas, color='tab:orange')
axes[1].set(xlabel='Training step', ylabel='Learned λ', title='Data-consistency weight')
plt.tight_layout(); plt.show()
print(f"Final logged loss: {losses[-1]:.6g}; final learned λ: {lambdas[-1]:.4f}")
"""
        ),
        markdown(
            """
## Validation checkpoint selection

Five evenly spaced checkpoints were evaluated on all 14 validation volumes. Metrics are
first averaged over slices within each volume, then averaged equally across volumes.
The test set did not participate in this choice.
"""
        ),
        code(
            """
validation = [
    {'Step': row['checkpoint_step'], **{key.upper(): value for key, value in row['modl'].items()}}
    for row in selection['candidates']
]
show_table(validation, [(key, key.replace('_', ' ').title()) for key in ['Step', 'PSNR', 'SSIM', 'NMSE', 'MEASUREMENT_RESIDUAL']])

fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
for axis, metric, better in zip(axes, ['PSNR', 'SSIM', 'NMSE'], ['higher', 'higher', 'lower']):
    axis.plot([row['Step'] for row in validation], [row[metric] for row in validation], marker='o')
    axis.axvline(selected_step, color='tab:green', linestyle='--')
    axis.set(xlabel='Checkpoint step', ylabel=metric, title=f'{metric} ({better} is better)')
plt.tight_layout(); plt.show()
print(f"Validation selected step {selected_step:,} by maximum volume-averaged PSNR.")
"""
        ),
        markdown("## Held-out test comparison"),
        code(
            """
comparison = []
for name, metrics in [('Zero-filled', test['zero_filled']), ('MoDL', test['modl']), ('RDBM', rdbm['rdbm'])]:
    comparison.append({
        'Method': name,
        'Complex NMSE ↓': metrics['nmse'],
        'Magnitude PSNR ↑': metrics['psnr'],
        'Magnitude SSIM ↑': metrics['ssim'],
        'Measurement residual ↓': metrics['measurement_residual'],
        'Seconds/slice ↓': metrics.get('seconds_per_slice', np.nan),
    })
show_table(comparison, [(key, key) for key in comparison[0]])

psnr_gain = test['modl']['psnr'] - test['zero_filled']['psnr']
ssim_change = test['modl']['ssim'] - test['zero_filled']['ssim']
speedup = rdbm['rdbm']['seconds_per_slice'] / test['modl']['seconds_per_slice']
print(f'MoDL vs zero-filled: {psnr_gain:+.2f} dB PSNR and {ssim_change:+.4f} SSIM.')
print(f'MoDL is approximately {speedup:.2f}× faster per slice than this RDBM evaluation.')
"""
        ),
        markdown(
            """
### Results by acquisition

This breakdown exposes behavior hidden by the overall mean. The small AXFLAIR subset
contains one strong complex-NMSE outlier, while both T1 groups improve in complex NMSE.
"""
        ),
        code(
            """
acquisition_rows = []
for acquisition, group in test['by_acquisition'].items():
    acquisition_rows.append({
        'Acquisition': acquisition,
        'Volumes': group['volume_count'],
        'ZF PSNR': group['zero_filled']['psnr'], 'MoDL PSNR': group['modl']['psnr'],
        'ZF SSIM': group['zero_filled']['ssim'], 'MoDL SSIM': group['modl']['ssim'],
        'ZF NMSE': group['zero_filled']['nmse'], 'MoDL NMSE': group['modl']['nmse'],
    })
acquisition = acquisition_rows
show_table(acquisition, [(key, key) for key in acquisition[0]])

x = np.arange(len(acquisition)); width = 0.36
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(x-width/2, [row['ZF PSNR'] for row in acquisition], width, label='Zero-filled')
axes[0].bar(x+width/2, [row['MoDL PSNR'] for row in acquisition], width, label='MoDL')
axes[0].set_xticks(x, [row['Acquisition'] for row in acquisition]); axes[0].set_ylabel('PSNR (dB)'); axes[0].legend()
axes[1].bar(x-width/2, [row['ZF NMSE'] for row in acquisition], width, label='Zero-filled')
axes[1].bar(x+width/2, [row['MoDL NMSE'] for row in acquisition], width, label='MoDL')
axes[1].set_xticks(x, [row['Acquisition'] for row in acquisition]); axes[1].set_ylabel('Complex NMSE'); axes[1].legend()
plt.tight_layout(); plt.show()
"""
        ),
        markdown("## Per-volume behavior"),
        code(
            """
volumes = [{
    'volume_id': item['volume_id'], 'acquisition': item['acquisition'],
    'ZF PSNR': item['zero_filled']['psnr'], 'MoDL PSNR': item['modl']['psnr'],
    'ZF NMSE': item['zero_filled']['nmse'], 'MoDL NMSE': item['modl']['nmse'],
} for item in test['volumes']]
for row in volumes:
    row['PSNR gain'] = row['MoDL PSNR'] - row['ZF PSNR']
show_table(sorted(volumes, key=lambda row: row['PSNR gain']), [(key, key) for key in volumes[0]])

fig, ax = plt.subplots(figsize=(6, 5))
for acquisition in sorted(set(row['acquisition'] for row in volumes)):
    group = [row for row in volumes if row['acquisition'] == acquisition]
    ax.scatter([row['ZF PSNR'] for row in group], [row['MoDL PSNR'] for row in group], label=acquisition, s=55)
all_psnr = [row[key] for row in volumes for key in ['ZF PSNR', 'MoDL PSNR']]
limits = [min(all_psnr) - .5, max(all_psnr) + .5]
ax.plot(limits, limits, '--', color='gray'); ax.set_xlim(limits); ax.set_ylim(limits)
ax.set(xlabel='Zero-filled PSNR', ylabel='MoDL PSNR', title='Each point is one test volume')
ax.legend(); plt.show()
"""
        ),
        markdown(
            """
## Reconstruction progression

The first row shows magnitude images; the second shows absolute error against the target.
The example was chosen automatically as the test slice closest to the median MoDL PSNR
gain, avoiding hand-picking the best-looking result.
"""
        ),
        code(
            """
print(f"Sample: {progression['sample_id']} (index {progression['sample_index']})")
display(Image(filename=str(ROOT / 'results/modl_evaluation/qualitative/unroll_progression.png')))
"""
        ),
        markdown("## Matched qualitative comparison"),
        code(
            """
display(Image(filename=str(ROOT / 'results/modl_evaluation/qualitative/model_comparison.png')))
metric_rows = [{'Method': name, **values} for name, values in progression['metrics'].items()]
show_table(metric_rows, [(key, key.replace('_', ' ').title()) for key in ['Method', 'nmse', 'psnr', 'ssim']])
"""
        ),
        markdown(
            """
## Interpretation

- Validation PSNR peaked at step **37,960**, so completing 50 epochs did not make the
  final checkpoint the best generalizing checkpoint.
- On the held-out test set, MoDL improves magnitude PSNR and measurement consistency
  over the zero-filled input. Its overall complex NMSE and magnitude SSIM are worse,
  driven partly by one AXFLAIR volume and by a systematic SSIM drop across acquisitions.
- RDBM is stronger overall on the present test metrics, while MoDL runs faster per slice.
- The next experiment should diagnose the AXFLAIR failure and revisit the MoDL loss and
  normalization before extending the comparison to 2×, 6×, and 8× acceleration.

These are first-run results on a small 14-volume test split, so they should be reported
with the acquisition breakdown rather than as a single aggregate score alone.
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
