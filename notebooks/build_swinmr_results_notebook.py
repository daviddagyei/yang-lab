"""Build a concise executed SwinMR all-acceleration results notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/swinmr_results.ipynb"


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
# Adapted complex SwinMR-PI at 2×, 4×, 6×, and 8×

This notebook reports four separately trained 50-epoch models. For each rate,
epochs 10, 20, 30, 40, and 50 were screened on validation data before one
evaluation of the selected checkpoint on the 14-volume held-out test split.

The official SwinMR backbone and composite image/frequency/perceptual objective
are retained. The implementation adapts the network to two-channel complex brain
MRI, supplied sensitivity maps, variable coil counts, and the project's frozen
Cartesian masks. It does **not** add a data-consistency layer.

Reference: [official repository](https://github.com/ayanglab/SwinMR)
"""
        ),
        code(
            """
from pathlib import Path
from html import escape
import json, os, re
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, Image, display

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DATA_ROOT = Path(os.environ.get('MRI_DATA_ROOT', '/data/david_agyei/nyu-dataset'))
OUTPUT_ROOT = Path(os.environ.get('SWINMR_OUTPUT_ROOT', DATA_ROOT/'derived'))
RATES = (2, 4, 6, 8)
def read_json(path): return json.loads(Path(path).read_text())
def artifacts(rate):
    run = OUTPUT_ROOT/f'swinmr_pi_complex_{rate}x'; evaluation = run/'evaluation'
    selection = read_json(evaluation/'best_validation_checkpoint.json')
    test_dir = evaluation/'test'/f"step-{selection['checkpoint_step']:07d}"
    return run, selection, test_dir, read_json(test_dir/'summary.json')
def show_table(rows, columns, precision=4, bold_best=None):
    bold_best = bold_best or {}; best = {}
    for key, direction in bold_best.items():
        values = [row[key] for row in rows if isinstance(row[key], (int,float)) and np.isfinite(row[key])]
        best[key] = (max if direction == 'max' else min)(values)
    def render(value): return f'{value:.{precision}f}' if isinstance(value,float) else escape(str(value))
    header = ''.join(f'<th style="text-align:left;padding:4px 10px">{escape(label)}</th>' for key,label in columns)
    def cell(row,key):
        value=render(row[key])
        if key in best and np.isclose(row[key],best[key]): value=f'<strong>{value}</strong>'
        return f'<td style="padding:4px 10px">{value}</td>'
    body=''.join('<tr>'+''.join(cell(row,key) for key,label in columns)+'</tr>' for row in rows)
    display(HTML(f'<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'))

runs = {rate: artifacts(rate) for rate in RATES}
figure_metadata = read_json(ROOT/'results/swinmr_evaluation/qualitative/all_rate_comparison.json')
"""
        ),
        md("## Validation selection"),
        code(
            """
rows=[]
fig,axes=plt.subplots(1,4,figsize=(15,3.2),sharey=True)
for axis,rate in zip(axes,RATES):
    selection=runs[rate][1]
    candidates=selection['candidates']
    rows.append({'Rate':f'{rate}×','Selected epoch':selection['completed_epochs'],
                 'Step':selection['checkpoint_step'],'Validation PSNR':selection['validation_metrics']['psnr'],
                 'Validation SSIM':selection['validation_metrics']['ssim']})
    axis.plot([x['completed_epochs'] for x in candidates],[x['swinmr']['psnr'] for x in candidates],marker='o')
    axis.axvline(selection['completed_epochs'],color='tab:green',linestyle='--')
    axis.set(xlabel='Epoch',title=f'{rate}×'); axis.set_ylabel('Validation PSNR (dB)')
show_table(rows,[(key,key) for key in rows[0]])
plt.tight_layout(); plt.show()
"""
        ),
        md("## Held-out test results"),
        code(
            """
rows=[]
for rate in RATES:
    test=runs[rate][3]; zf=test['zero_filled']; result=test['swinmr']
    rows.append({'Rate':f'{rate}×','ZF PSNR':zf['psnr'],'SwinMR PSNR':result['psnr'],
                 'PSNR gain':result['psnr']-zf['psnr'],'ZF SSIM':zf['ssim'],
                 'SwinMR SSIM':result['ssim'],'Complex NMSE':result['nmse'],
                 'Residual':result['measurement_residual'],'Seconds/slice':result['seconds_per_slice']})
show_table(rows,[(key,key) for key in rows[0]])
fig,axes=plt.subplots(1,3,figsize=(12,3.4))
axes[0].plot(RATES,[row['SwinMR PSNR'] for row in rows],marker='o'); axes[0].set(title='PSNR',xlabel='Acceleration',ylabel='dB')
axes[1].plot(RATES,[row['SwinMR SSIM'] for row in rows],marker='o'); axes[1].set(title='SSIM',xlabel='Acceleration')
axes[2].plot(RATES,[row['Complex NMSE'] for row in rows],marker='o'); axes[2].set(title='Complex NMSE',xlabel='Acceleration')
plt.tight_layout(); plt.show()
"""
        ),
        md("## Training trajectories"),
        code(
            """
pattern=re.compile(r'epoch_complete=(\d+)/\d+ step=(\d+) mean_loss=([0-9.eE+-]+) mean_coil_image_charbonnier=([0-9.eE+-]+) mean_coil_kspace_charbonnier=([0-9.eE+-]+) mean_perceptual=([0-9.eE+-]+)')
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
for rate in RATES:
    run=runs[rate][0]
    records=[match.groups() for line in (run/'train.log').read_text().splitlines() if (match:=pattern.search(line))]
    records=list({int(row[0]):row for row in records}.values()); records.sort(key=lambda row:int(row[0]))
    epochs=[int(row[0]) for row in records]
    axes[0].plot(epochs,[float(row[2]) for row in records],label=f'{rate}×')
    axes[1].plot(epochs,[float(row[4]) for row in records],label=f'{rate}×')
axes[0].set(xlabel='Epoch',ylabel='Mean composite loss',title='Training loss')
axes[1].set(xlabel='Epoch',ylabel='Mean frequency loss',title='Frequency component')
for axis in axes: axis.legend()
plt.tight_layout(); plt.show()
"""
        ),
        md("## Matched reconstruction comparison"),
        code(
            """
print(f"Sample: {figure_metadata['sample_id']} ({figure_metadata['selection_reason']})")
display(Image(filename=str(ROOT/'results/swinmr_evaluation/qualitative/all_rate_comparison.png'),width=3000))
rows=[{'State':name,'PSNR':values['psnr'],'SSIM':values['ssim'],'NMSE':values['nmse']}
      for name,values in figure_metadata['metrics'].items()]
show_table(rows,[(key,key) for key in rows[0]])
"""
        ),
        md(
            """
## Interpretation

- Each acceleration rate uses independently trained weights and independently
  selected validation checkpoints.
- Quality is expected to decrease as acceleration rises because fewer phase-encoding
  measurements are retained.
- SwinMR has no explicit acquired-sample replacement or CG-SENSE layer; its reported
  residual therefore measures a genuine architectural tradeoff.
- Timing is system-dependent and is descriptive of this two-GPU workstation.
"""
        ),
        md("## Reproducibility artifacts"),
        code(
            """
rows=[]
for rate in RATES:
    run,selection,test_dir,test=runs[rate]
    rows.extend([
        {'Rate':f'{rate}×','Artifact':'Selected checkpoint','Path':selection['checkpoint']},
        {'Rate':f'{rate}×','Artifact':'Test summary','Path':str(test_dir/'summary.json')},
        {'Rate':f'{rate}×','Artifact':'Slice metrics','Path':str(test_dir/'slice_metrics.jsonl')},
    ])
show_table(rows,[(key,key) for key in rows[0]],precision=0)
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
