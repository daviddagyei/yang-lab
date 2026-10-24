"""Build a concise executed-results notebook for the adapted D5-C5 model."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/cascade_cnn_results.ipynb"


def md(value: str):
    return nbf.v4.new_markdown_cell(value.strip())


def code(value: str):
    return nbf.v4.new_code_cell(value.strip())


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
        md(
            """
# Adapted multicoil D5-C5 Cascade CNN at 4×

This notebook reports the completed **50-epoch / 9,700-update** experiment.
Epochs 10, 20, 30, 40, and 50 were screened on validation data, and the selected
checkpoint was evaluated once on the held-out test split.

The five independent D5 residual CNN cascades match the published architecture.
The original single-image Fourier replacement is replaced by a fixed 10-iteration
CG-SENSE proximal solve for raw multicoil measurements, so this is an **adapted
multicoil D5-C5**, rather than an exact reproduction.

References: [paper](https://arxiv.org/abs/1703.00555) ·
[official repository](https://github.com/js3611/Deep-MRI-Reconstruction)
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
RUN = Path('/data/david_agyei/nyu-dataset/derived/multicoil_cascade_cnn_4x')
EVAL = RUN / 'evaluation'

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
test_dir = EVAL / 'test' / f'step-{step:07d}'
test = read_json(test_dir / 'summary.json')
test_slices = [json.loads(line) for line in (test_dir/'slice_metrics.jsonl').read_text().splitlines() if line]
rdbm = read_json(Path('/data/david_agyei/nyu-dataset/derived/rdbm_full_bf16/evaluation/test/step-0094900/summary.json'))
unrolled = read_json(Path('/data/david_agyei/nyu-dataset/derived/unrolled_rdbm_4x_scratch/evaluation/test/step-0037960/summary.json'))
modl = read_json(Path('/data/david_agyei/nyu-dataset/derived/modl_4x_fp32/evaluation/test/step-0037960/summary.json'))
progression = read_json(ROOT/'results/cascade_cnn_evaluation/qualitative/cascade_progression.json')
print(f"Selected epoch {selection['completed_epochs']}, step {step:,}")
"""
        ),
        md("## Training trajectory"),
        code(
            """
pattern = re.compile(
    r'epoch_complete=(\d+)/\d+ step=(\d+) mean_loss=([0-9.eE+-]+) '
    r'mean_complex_mse=([0-9.eE+-]+) mean_l2_regularization=([0-9.eE+-]+)'
)
records = [match.groups() for line in (RUN/'train.log').read_text().splitlines() if (match := pattern.search(line))]
records_by_epoch = {int(row[0]): row for row in records}
records = [records_by_epoch[epoch] for epoch in sorted(records_by_epoch)]
epochs = np.array([int(row[0]) for row in records])
loss = np.array([float(row[2]) for row in records])
complex_mse = np.array([float(row[3]) for row in records])
l2 = np.array([float(row[4]) for row in records])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(epochs, loss, label='Total'); axes[0].plot(epochs, complex_mse, label='Complex MSE')
axes[0].set(xlabel='Epoch', ylabel='Epoch mean', title='Training objectives'); axes[0].legend()
axes[1].plot(epochs, l2, color='tab:orange'); axes[1].set(xlabel='Epoch', ylabel='L2 term', title='Weight regularization')
plt.tight_layout(); plt.show()
print(f'Final mean loss: {loss[-1]:.6f}; complex MSE: {complex_mse[-1]:.6f}.')
"""
        ),
        md("## Validation checkpoint selection"),
        code(
            """
validation = [{'Epoch': row['completed_epochs'], 'Step': row['checkpoint_step'],
               'PSNR ↑': row['cascade_cnn']['psnr'], 'SSIM ↑': row['cascade_cnn']['ssim'],
               'NMSE ↓': row['cascade_cnn']['nmse'], 'Residual ↓': row['cascade_cnn']['measurement_residual']}
              for row in selection['candidates']]
show_table(validation, [(key, key) for key in validation[0]],
           bold_best={'PSNR ↑':'max', 'SSIM ↑':'max', 'NMSE ↓':'min', 'Residual ↓':'min'})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
for axis, metric in zip(axes, ['PSNR ↑', 'SSIM ↑', 'NMSE ↓']):
    axis.plot([row['Epoch'] for row in validation], [row[metric] for row in validation], marker='o')
    axis.axvline(selection['completed_epochs'], color='tab:green', linestyle='--')
    axis.set(xlabel='Epoch', title=metric)
plt.tight_layout(); plt.show()
print(f"Epoch {selection['completed_epochs']} selected by maximum volume-averaged validation PSNR.")
"""
        ),
        md("## Held-out test comparison"),
        code(
            """
comparison = []
for name, values in [
    ('Zero-filled', test['zero_filled']), ('MoDL', modl['modl']),
    ('RDBM', rdbm['rdbm']), ('Unrolled RDBM', unrolled['unrolled_rdbm']),
    ('Cascade CNN', test['cascade_cnn'])
]:
    comparison.append({'Method': name, 'PSNR ↑': values['psnr'], 'SSIM ↑': values['ssim'],
                       'Complex NMSE ↓': values['nmse'], 'Residual ↓': values['measurement_residual'],
                       'Seconds/slice ↓': values.get('seconds_per_slice', np.nan)})
show_table(comparison, [(key, key) for key in comparison[0]],
           bold_best={'PSNR ↑':'max', 'SSIM ↑':'max', 'Complex NMSE ↓':'min',
                      'Residual ↓':'min', 'Seconds/slice ↓':'min'})

result = test['cascade_cnn']; baseline = test['zero_filled']
print(f"Cascade vs zero-filled: {result['psnr']-baseline['psnr']:+.2f} dB PSNR, "
      f"{result['ssim']-baseline['ssim']:+.4f} SSIM, and "
      f"{100*(1-result['nmse']/baseline['nmse']):.1f}% lower complex NMSE.")
print(f"Cascade vs RDBM: {result['psnr']-rdbm['rdbm']['psnr']:+.2f} dB PSNR and "
      f"{100*(1-result['nmse']/rdbm['rdbm']['nmse']):.1f}% lower complex NMSE.")
"""
        ),
        md("## Results by acquisition"),
        code(
            """
acquisition = []
for name, group in test['by_acquisition'].items():
    acquisition.append({'Acquisition': name, 'Volumes': group['volume_count'],
        'ZF PSNR': group['zero_filled']['psnr'], 'Cascade PSNR': group['cascade_cnn']['psnr'],
        'ZF SSIM': group['zero_filled']['ssim'], 'Cascade SSIM': group['cascade_cnn']['ssim'],
        'ZF NMSE': group['zero_filled']['nmse'], 'Cascade NMSE': group['cascade_cnn']['nmse']})
show_table(acquisition, [(key, key) for key in acquisition[0]])

x=np.arange(len(acquisition)); width=.36
fig, axes=plt.subplots(1,2,figsize=(12,4))
axes[0].bar(x-width/2,[row['ZF PSNR'] for row in acquisition],width,label='Zero-filled')
axes[0].bar(x+width/2,[row['Cascade PSNR'] for row in acquisition],width,label='Cascade CNN')
axes[0].set_xticks(x,[row['Acquisition'] for row in acquisition]); axes[0].set_ylabel('PSNR (dB)'); axes[0].legend()
axes[1].bar(x-width/2,[row['ZF NMSE'] for row in acquisition],width,label='Zero-filled')
axes[1].bar(x+width/2,[row['Cascade NMSE'] for row in acquisition],width,label='Cascade CNN')
axes[1].set_xticks(x,[row['Acquisition'] for row in acquisition]); axes[1].set_ylabel('Complex NMSE'); axes[1].legend()
plt.tight_layout(); plt.show()
"""
        ),
        md(
            """
## Reconstruction across cascades

The example is selected automatically as the test slice closest to the median
Cascade CNN PSNR gain. Image and error scales are shared across columns. The RDBM
output uses a fixed random seed for reproducibility.
"""
        ),
        code(
            """
print(f"Sample: {progression['sample_id']} (index {progression['sample_index']})")
display(Image(filename=str(ROOT/'results/cascade_cnn_evaluation/qualitative/cascade_progression.png'), width=2400))
metric_rows=[{'State':name,'PSNR':values['psnr'],'SSIM':values['ssim'],'NMSE':values['nmse']}
             for name,values in progression['metrics'].items()]
show_table(metric_rows, [(key,key) for key in metric_rows[0]])
"""
        ),
        md(
            """
## Interpretation

- The selected Cascade CNN improves all reported image-quality metrics over the
  zero-filled reconstruction.
- It is slightly below the original RDBM in PSNR and SSIM, while producing lower
  complex NMSE and faster inference in these evaluations.
- Epoch 50 gives the highest validation PSNR and lowest validation NMSE. Epoch 40
  has marginally higher validation SSIM, but PSNR was the predeclared selection metric.
- The aggregate complex NMSE is strongly affected by one AXFLAIR failure case;
  both T1 acquisition groups improve substantially over zero-filled reconstruction.
- Timing is descriptive and should be remeasured in isolated repeated runs before
  making formal speed claims.
"""
        ),
        md("## Reproducibility artifacts"),
        code(
            """
show_table([
    {'Artifact':'Selected checkpoint','Path':selection['checkpoint']},
    {'Artifact':'Checkpoint selection','Path':str(EVAL/'best_validation_checkpoint.json')},
    {'Artifact':'Test summary','Path':str(test_dir/'summary.json')},
    {'Artifact':'Test slice metrics','Path':str(test_dir/'slice_metrics.jsonl')},
    {'Artifact':'Training log','Path':str(RUN/'train.log')},
], [('Artifact','Artifact'),('Path','Path')], precision=0)
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
