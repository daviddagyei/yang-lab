"""Build the results notebook for the target-free ASSCGD-inspired 4x experiment."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "asscgd_results.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        markdown(
            """
# Target-free ASSCGD-inspired 4× MRI reconstruction

This notebook records the completed **50-epoch / 94,900-step** self-supervised
diffusion experiment. Five checkpoints were screened on validation data, and the
selected checkpoint was evaluated once on the held-out test set. The validation
screen covers epochs **10, 20, 30, 40, and 50**.

The implementation is **ASSCGD/SSAD-MRI-inspired**, rather than a claim of exact
reproduction: it keeps this project's conditional U-Net, uses supplied sensitivity
maps, and combines held-out acquired k-space supervision with adversarial diffusion
training. Fully sampled images were never used to optimize the model; they are loaded
only for validation checkpoint selection and final evaluation.

References: [paper preprint](https://arxiv.org/abs/2406.15656) ·
[official ReconFormer repository](https://github.com/guopengf/ReconFormer)
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
ASSCGD_ROOT = Path('/data/david_agyei/nyu-dataset/derived/asscgd_4x_stable')
EVAL = ASSCGD_ROOT / 'evaluation'

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
test_dir = EVAL / 'test' / f'step-{selected_step:07d}'
test = read_json(test_dir / 'summary.json')
test_slices = [json.loads(line) for line in (test_dir / 'slice_metrics.jsonl').read_text().splitlines() if line]
progression = read_json(ROOT / 'results/asscgd_evaluation/qualitative/progression_metadata.json')
print(f"Selected checkpoint: epoch {selection['completed_epochs']}, step {selected_step:,}")
"""
        ),
        markdown(
            """
## Protocol and data integrity

Splits are frozen at the volume level, preventing slices from the same volume from
appearing in multiple splits. Training dynamically divides each acquired 4× mask into
input and held-out lines while always retaining ACS lines in the input. The loss uses
only held-out measured k-space; validation and test targets are evaluation-only.
"""
        ),
        code(
            """
integrity = [
    {'Split': 'Train', 'Volumes': 120, 'Slices': 1898, 'Role': 'target-free optimization'},
    {'Split': 'Validation', 'Volumes': 14, 'Slices': 220, 'Role': 'checkpoint selection'},
    {'Split': 'Test', 'Volumes': test['volume_count'], 'Slices': test['slice_count'], 'Role': 'final reporting'},
]
show_table(integrity, [(key, key) for key in ['Split', 'Volumes', 'Slices', 'Role']], precision=0)
protocol = [
    {'Setting': 'Acceleration', 'Value': '4× Cartesian'},
    {'Setting': 'Sensitivity maps', 'Value': 'supplied maps only'},
    {'Setting': 'Training target', 'Value': 'held-out acquired k-space'},
    {'Setting': 'Diffusion schedule', 'Value': '100 train / 10 sampling steps'},
    {'Setting': 'Inference noise seed', 'Value': test['noise_seed']},
    {'Setting': 'Runtime measurement', 'Value': 'shared GPU; descriptive only'},
]
show_table(protocol, [('Setting', 'Setting'), ('Value', 'Value')], precision=0)
"""
        ),
        markdown("## Training stability"),
        code(
            """
pattern = re.compile(
    r'epoch_complete=(\d+)/\d+ step=(\d+) mean_loss=([0-9.eE+-]+) '
    r'mean_heldout=([0-9.eE+-]+) mean_g_adv=([0-9.eE+-]+) mean_d=([0-9.eE+-]+)'
)
records = []
for line in (ASSCGD_ROOT / 'train.log').read_text().splitlines():
    match = pattern.search(line)
    if match:
        records.append(tuple(float(value) for value in match.groups()))
records = sorted({int(row[0]): row for row in records}.values())
epochs = np.array([int(row[0]) for row in records])
loss = np.array([row[2] for row in records])
heldout = np.array([row[3] for row in records])
g_adv = np.array([row[4] for row in records])
disc = np.array([row[5] for row in records])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs, loss, marker='o', label='total generator loss')
axes[0].plot(epochs, heldout, marker='o', label='held-out k-space')
axes[0].set(xlabel='Completed epoch', ylabel='Epoch mean', title='Generator objectives')
axes[0].legend()
axes[1].plot(epochs, g_adv, marker='o', label='generator adversarial')
axes[1].plot(epochs, disc, marker='o', label='discriminator')
axes[1].set(xlabel='Completed epoch', ylabel='Epoch mean', title='Adversarial balance')
axes[1].legend()
plt.tight_layout(); plt.show()
print(f'Logged stabilized run: epochs {epochs[0]}–{epochs[-1]}.')
print(f'Final means: loss={loss[-1]:.4f}, held-out={heldout[-1]:.4f}, '
      f'generator adversarial={g_adv[-1]:.4f}, discriminator={disc[-1]:.4f}.')
"""
        ),
        markdown(
            """
The generator recovered from the epoch-2 checkpoint with the stabilized discriminator
configuration. The stable log therefore starts at epoch 3. Held-out k-space loss falls
gradually while discriminator loss remains near 1.376, avoiding the earlier collapsed
discriminator regime.
"""
        ),
        markdown("## Validation checkpoint selection"),
        code(
            """
validation = [{
    'Epoch': row['completed_epochs'], 'Step': row['checkpoint_step'],
    'PSNR ↑': row['asscgd']['psnr'], 'SSIM ↑': row['asscgd']['ssim'],
    'Complex NMSE ↓': row['asscgd']['nmse'],
    'Residual ↓': row['asscgd']['measurement_residual'],
} for row in selection['candidates']]
show_table(validation, [(key, key) for key in validation[0]])

fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
for axis, metric in zip(axes, ['PSNR ↑', 'SSIM ↑', 'Complex NMSE ↓']):
    axis.plot([row['Epoch'] for row in validation], [row[metric] for row in validation], marker='o')
    axis.axvline(selection['completed_epochs'], color='tab:green', linestyle='--')
    axis.set(xlabel='Epoch', ylabel=metric, title=metric)
plt.tight_layout(); plt.show()
print(f"Epoch {selection['completed_epochs']} was selected by maximum volume-averaged validation PSNR.")
"""
        ),
        markdown("## Held-out test result"),
        code(
            """
comparison = []
for name, values in [('Zero-filled', test['zero_filled']), ('ASSCGD-inspired', test['asscgd'])]:
    comparison.append({
        'Method': name, 'Complex NMSE ↓': values['nmse'],
        'Magnitude PSNR ↑': values['psnr'], 'Magnitude SSIM ↑': values['ssim'],
        'Residual ↓': values['measurement_residual'],
        'Seconds/slice† ↓': values.get('seconds_per_slice', np.nan),
    })
show_table(comparison, [(key, key) for key in comparison[0]])

psnr_gain = test['asscgd']['psnr'] - test['zero_filled']['psnr']
ssim_change = test['asscgd']['ssim'] - test['zero_filled']['ssim']
nmse_reduction = 100 * (1 - test['asscgd']['nmse'] / test['zero_filled']['nmse'])
residual_reduction = 100 * (1 - test['asscgd']['measurement_residual'] / test['zero_filled']['measurement_residual'])
print(f'ASSCGD vs zero-filled: {psnr_gain:+.2f} dB PSNR, {ssim_change:+.4f} SSIM, '
      f'{nmse_reduction:.1f}% lower NMSE, and {residual_reduction:.1f}% lower measurement residual.')
"""
        ),
        markdown(
            """
† The final checkpoint was evaluated while the unrolled-RDBM experiment was training
on the same GPU. Reconstruction time therefore describes this run under shared load;
it is not an isolated speed benchmark. Image-quality metrics are unaffected.
"""
        ),
        markdown("### Results by acquisition"),
        code(
            """
acquisition = []
for name, group in test['by_acquisition'].items():
    acquisition.append({
        'Acquisition': name, 'Volumes': group['volume_count'],
        'ZF PSNR': group['zero_filled']['psnr'], 'ASSCGD PSNR': group['asscgd']['psnr'],
        'ZF SSIM': group['zero_filled']['ssim'], 'ASSCGD SSIM': group['asscgd']['ssim'],
        'ZF NMSE': group['zero_filled']['nmse'], 'ASSCGD NMSE': group['asscgd']['nmse'],
    })
show_table(acquisition, [(key, key) for key in acquisition[0]])

x = np.arange(len(acquisition)); width = .36
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(x-width/2, [row['ZF PSNR'] for row in acquisition], width, label='Zero-filled')
axes[0].bar(x+width/2, [row['ASSCGD PSNR'] for row in acquisition], width, label='ASSCGD')
axes[0].set_xticks(x, [row['Acquisition'] for row in acquisition]); axes[0].set_ylabel('PSNR (dB)'); axes[0].legend()
axes[1].bar(x-width/2, [row['ZF SSIM'] for row in acquisition], width, label='Zero-filled')
axes[1].bar(x+width/2, [row['ASSCGD SSIM'] for row in acquisition], width, label='ASSCGD')
axes[1].set_xticks(x, [row['Acquisition'] for row in acquisition]); axes[1].set_ylabel('SSIM'); axes[1].legend()
plt.tight_layout(); plt.show()
"""
        ),
        markdown("## Slice- and volume-level behavior"),
        code(
            """
slice_psnr_gain = np.array([row['asscgd_psnr'] - row['zero_filled_psnr'] for row in test_slices])
slice_ssim_gain = np.array([row['asscgd_ssim'] - row['zero_filled_ssim'] for row in test_slices])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(slice_psnr_gain, bins=30, color='tab:blue'); axes[0].axvline(0, color='black', linestyle='--')
axes[0].set(xlabel='ASSCGD − zero-filled PSNR (dB)', ylabel='Slices', title='Per-slice PSNR change')
axes[1].hist(slice_ssim_gain, bins=30, color='tab:orange'); axes[1].axvline(0, color='black', linestyle='--')
axes[1].set(xlabel='ASSCGD − zero-filled SSIM', ylabel='Slices', title='Per-slice SSIM change')
plt.tight_layout(); plt.show()
print(f'PSNR improves on {100*np.mean(slice_psnr_gain > 0):.1f}% of test slices; '
      f'SSIM improves on {100*np.mean(slice_ssim_gain > 0):.1f}%.')

volumes = [{
    'Volume': item['volume_id'], 'Acquisition': item['acquisition'],
    'ZF PSNR': item['zero_filled']['psnr'], 'ASSCGD PSNR': item['asscgd']['psnr'],
    'PSNR gain': item['asscgd']['psnr'] - item['zero_filled']['psnr'],
    'ZF SSIM': item['zero_filled']['ssim'], 'ASSCGD SSIM': item['asscgd']['ssim'],
} for item in test['volumes']]
show_table(sorted(volumes, key=lambda row: row['PSNR gain']), [(key, key) for key in volumes[0]])
"""
        ),
        markdown(
            """
## Reconstruction across training checkpoints

The example is selected automatically as the test slice closest to the median ASSCGD
PSNR gain. All checkpoints use the same deterministic inference noise. The top row is
magnitude; the lower row is absolute magnitude error against the fully sampled target.
"""
        ),
        code(
            """
print(f"Sample: {progression['sample_id']} (index {progression['sample_index']})")
display(Image(filename=str(ROOT / 'results/asscgd_evaluation/qualitative/checkpoint_progression.png')))
"""
        ),
        markdown("## Reverse-diffusion reconstruction timeline"),
        code(
            """
display(Image(filename=str(ROOT / 'results/asscgd_evaluation/qualitative/reverse_diffusion_progression.png')))
metric_rows = []
for name, values in progression['metrics'].items():
    metric_rows.append({'State': name, 'NMSE': values['nmse'], 'PSNR': values['psnr'], 'SSIM': values['ssim']})
show_table(metric_rows, [(key, key) for key in metric_rows[0]])
"""
        ),
        markdown(
            """
## Interpretation and limitations

- The self-supervised model improves test magnitude PSNR, complex NMSE, and acquired-line
  consistency over the zero-filled reconstruction without using fully sampled training targets.
- Magnitude SSIM decreases overall. The model sharpens the reconstruction and reduces global
  squared error, but its local structural contrast does not yet match the target reliably.
- AXFLAIR contains only two test volumes and has substantially worse complex NMSE than the T1
  acquisitions, so the aggregate should always be reported with this stratified breakdown.
- Evaluation uses one deterministic noise realization per slice. This makes checkpoint comparison
  and reruns reproducible, but it does not estimate uncertainty across stochastic samples.
- Reconstruction timing was measured under concurrent GPU load and should not be used for a
  direct speed comparison with models benchmarked on an otherwise idle GPU.
- This is an ASSCGD/SSAD-MRI-inspired adaptation using the existing conditional U-Net and supplied
  sensitivity maps. Architecture and dataset differences prevent direct numerical comparison with
  the paper's reported results.

The clearest next experiment is to improve the image-domain objective or architecture with SSIM
as a validation constraint, while retaining the target-free held-out k-space protocol.
"""
        ),
        markdown("## Reproducibility artifacts"),
        code(
            """
artifacts = [
    {'Artifact': 'Checkpoint selection', 'Path': str(EVAL / 'best_validation_checkpoint.json')},
    {'Artifact': 'Test summary', 'Path': str(test_dir / 'summary.json')},
    {'Artifact': 'Test slice metrics', 'Path': str(test_dir / 'slice_metrics.jsonl')},
    {'Artifact': 'Training log', 'Path': str(ASSCGD_ROOT / 'train.log')},
    {'Artifact': 'Progression metadata', 'Path': str(ROOT / 'results/asscgd_evaluation/qualitative/progression_metadata.json')},
]
show_table(artifacts, [('Artifact', 'Artifact'), ('Path', 'Path')], precision=0)
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
