"""Build a concise executed-results source notebook for the unrolled RDBM."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "unrolled_rdbm_results.ipynb"


def md(value: str):
    return nbf.v4.new_markdown_cell(value.strip())


def code(value: str):
    return nbf.v4.new_code_cell(value.strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        md(
            """
# Unrolled RDBM 4× reconstruction

This notebook reviews the **50-epoch / 94,900-step** supervised experiment trained
from random initialization. The model shares one RDBM-derived denoiser across three
cascades and applies a six-iteration conjugate-gradient SENSE consistency solve after
each cascade. Epochs 10, 20, 30, 40, and 50 were screened on validation data; the
selected checkpoint was then evaluated on the held-out test set.
"""
        ),
        code(
            """
from pathlib import Path
from html import escape
import json, re
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, Image, display

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
RUN = Path('/data/david_agyei/nyu-dataset/derived/unrolled_rdbm_4x_scratch')
EVAL = RUN / 'evaluation'
RDBM_EVAL = Path('/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16/evaluation')

def read_json(path): return json.loads(Path(path).read_text())
def show_table(rows, columns, precision=4, bold_best=None):
    bold_best = bold_best or {}
    best = {}
    for key, direction in bold_best.items():
        values = [row[key] for row in rows if isinstance(row[key], (int, float)) and np.isfinite(row[key])]
        best[key] = (max if direction == 'max' else min)(values)
    def render(value): return f'{value:.{precision}f}' if isinstance(value, float) else escape(str(value))
    header = ''.join(f'<th style="text-align:left;padding:4px 10px">{escape(label)}</th>' for key, label in columns)
    def cell(row, key):
        value = render(row[key])
        if key in best and np.isclose(row[key], best[key]): value = f'<strong>{value}</strong>'
        return f'<td style="padding:4px 10px">{value}</td>'
    body = ''.join('<tr>' + ''.join(cell(row, key) for key, label in columns) + '</tr>' for row in rows)
    display(HTML(f'<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'))

selection = read_json(EVAL / 'best_validation_checkpoint.json')
step = selection['checkpoint_step']
test = read_json(EVAL / 'test' / f'step-{step:07d}' / 'summary.json')
rdbm = read_json(RDBM_EVAL / 'test/step-0094900/summary.json')
figure_metadata = read_json(ROOT / 'results/unrolled_rdbm_evaluation/qualitative/cascade_comparison.json')
print(f"Selected epoch {selection['completed_epochs']}, step {step:,}")
"""
        ),
        md("## Training trajectory"),
        code(
            """
pattern = re.compile(r'epoch_complete=(\d+)/\d+ step=(\d+) mean_loss=([0-9.eE+-]+).*?mean_ssim_loss=([0-9.eE+-]+) mean_measurement=([0-9.eE+-]+)')
records = [match.groups() for line in (RUN/'train.log').read_text().splitlines() if (match := pattern.search(line))]
records = sorted({int(row[0]): row for row in records}.values())
epochs = np.array([int(row[0]) for row in records])
loss = np.array([float(row[2]) for row in records])
ssim_loss = np.array([float(row[3]) for row in records])
measurement = np.array([float(row[4]) for row in records])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(epochs, loss); axes[0].set(xlabel='Epoch', ylabel='Mean loss', title='Training loss')
axes[1].plot(epochs, ssim_loss, label='SSIM loss'); axes[1].plot(epochs, measurement, label='Measurement loss')
axes[1].set(xlabel='Epoch', title='Training components'); axes[1].legend()
plt.tight_layout(); plt.show()
print(f'Final epoch mean loss: {loss[-1]:.4f}')
"""
        ),
        md("## Validation checkpoint selection"),
        code(
            """
validation = [{'Epoch': row['completed_epochs'], 'Step': row['checkpoint_step'],
               'PSNR ↑': row['unrolled_rdbm']['psnr'], 'SSIM ↑': row['unrolled_rdbm']['ssim'],
               'NMSE ↓': row['unrolled_rdbm']['nmse'], 'Residual ↓': row['unrolled_rdbm']['measurement_residual']}
              for row in selection['candidates']]
show_table(validation, [(key, key) for key in validation[0]])
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
for axis, metric in zip(axes, ['PSNR ↑', 'SSIM ↑', 'NMSE ↓']):
    axis.plot([row['Epoch'] for row in validation], [row[metric] for row in validation], marker='o')
    axis.axvline(selection['completed_epochs'], color='tab:green', linestyle='--')
    axis.set(xlabel='Epoch', title=metric)
plt.tight_layout(); plt.show()
print(f"Epoch {selection['completed_epochs']} selected by validation PSNR.")
"""
        ),
        md("## Held-out test comparison"),
        code(
            """
comparison = []
for name, values in [('Zero-filled', test['zero_filled']), ('RDBM', rdbm['rdbm']), ('Unrolled RDBM', test['unrolled_rdbm'])]:
    comparison.append({'Method': name, 'PSNR ↑': values['psnr'], 'SSIM ↑': values['ssim'],
                       'Complex NMSE ↓': values['nmse'], 'Residual ↓': values['measurement_residual'],
                       'Seconds/slice ↓': values.get('seconds_per_slice', np.nan)})
show_table(
    comparison,
    [(key, key) for key in comparison[0]],
    bold_best={'PSNR ↑': 'max', 'SSIM ↑': 'max', 'Complex NMSE ↓': 'min',
               'Residual ↓': 'min', 'Seconds/slice ↓': 'min'},
)
gain = test['unrolled_rdbm']['psnr'] - rdbm['rdbm']['psnr']
nmse_change = 100 * (1 - test['unrolled_rdbm']['nmse'] / rdbm['rdbm']['nmse'])
print(f'Unrolled vs RDBM: {gain:+.2f} dB PSNR and {nmse_change:.1f}% lower complex NMSE.')
"""
        ),
        md(
            """
## Representative reconstruction and cascade progression

The slice is selected automatically as the test case closest to the median unrolled
PSNR gain. Image and error scales are shared across columns.
"""
        ),
        code(
            """
print(f"Sample: {figure_metadata['sample_id']}")
display(Image(filename=str(ROOT/'results/unrolled_rdbm_evaluation/qualitative/cascade_comparison.png'), width=2400))
"""
        ),
        md(
            """
## Review notes

- The validation-selected unrolled model improves PSNR and complex NMSE over the
  original RDBM on the same held-out test volumes.
- RDBM retains higher SSIM and a slightly lower measurement residual, showing that
  the unrolled objective favors global error reduction more than structural similarity.
- Validation peaks at epoch 20; continued optimization does not improve validation
  PSNR, so epoch 20 is reported instead of the final epoch.
- Inference timing is descriptive; formal speed claims require repeated isolated runs.
"""
        ),
        md("## Artifacts"),
        code(
            """
show_table([
    {'Artifact': 'Checkpoint selection', 'Path': str(EVAL/'best_validation_checkpoint.json')},
    {'Artifact': 'Test summary', 'Path': str(EVAL/'test'/f'step-{step:07d}'/'summary.json')},
    {'Artifact': 'Test slice metrics', 'Path': str(EVAL/'test'/f'step-{step:07d}'/'slice_metrics.jsonl')},
    {'Artifact': 'Training log', 'Path': str(RUN/'train.log')},
], [('Artifact', 'Artifact'), ('Path', 'Path')], precision=0)
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
