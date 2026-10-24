# ASSCGD-inspired self-supervised MRI reconstruction

This package starts the project's 4x self-supervised reconstruction track. It
reuses the frozen 148-volume supplied-sensitivity-map cohort and the existing
multi-coil operator `A(x) = M F(Sx)`.

For each slice, the existing 4x acquired mask `Omega` is divided into two
disjoint sets:

- `Theta` (`input_mask`) supplies the network input and future data-consistency
  operations;
- `Lambda` (`loss_mask`) supplies the measurement-domain training loss;
- the auto-calibration center remains entirely in `Theta`;
- `Theta union Lambda = Omega`, and `Theta intersection Lambda` is empty.

Training partitions change deterministically with `dataset.set_epoch(epoch)`.
Validation and test partitions remain fixed at epoch zero. The loader never reads
or returns `reconstruction_rss` or a fully sampled complex target. Its scale and
zero-filled initialization use only `Theta`, preventing held-out measurements from
affecting the model input.

The generator uses the project's conditional U-Net with a standard
cosine diffusion schedule. At training time it corrupts the input-only initial
reconstruction, predicts a clean complex image, applies multi-coil consistency on
`Theta`, and evaluates the reconstruction only against `Lambda`. Inference uses
all acquired 4x measurements and the paper's stated initial noise variance of 0.1.

The adversarial mapper conditions a four-convolution discriminator on a later
noisy diffusion state. Its convolutions use spectral normalization, and it trains
once per four generator updates at one tenth of the generator learning rate. Real
label smoothing and an adaptive cap keep the adversarial term at no more than 10%
of the held-out loss. These stability controls preserve the paper's nominal
adversarial weight of 0.1 while preventing the discriminator from overwhelming
the reconstruction objective. The discriminator never receives held-out k-space
or a fully sampled target. Training stops before saving an epoch if the mean
discriminator loss falls below 0.2.

This is an **ASSCGD/SSAD-MRI-inspired adaptation**, rather than a claim of exact
reproduction. The paper uses ReconFormer as its generator, whereas this project
retains its existing conditional U-Net to isolate the effect of self-supervision.
The authors' paper specifies the framework but no authors' implementation was
publicly linked when this package was written.

Primary references:

- Safari et al., *Self-supervised adversarial diffusion models for fast MRI
  reconstruction*: https://doi.org/10.1002/mp.17675
- Authors' preprint (ASSCGD terminology): https://arxiv.org/abs/2406.15656
- Yaman et al., *Self-supervised learning of physics-guided reconstruction neural
  networks without fully sampled reference data* (SSDU):
  https://doi.org/10.1002/mrm.28378
- Official ReconFormer generator repository referenced by the ASSCGD architecture:
  https://github.com/guopengf/ReconFormer

The project keeps its established 8% calibration center for controlled comparison
with the completed RDBM and MoDL runs. Changing that mask would create a different
4x experiment.

Verification sequence:

```bash
bash scripts/check_asscgd_data.sh
bash scripts/smoke_asscgd_model.sh
bash scripts/overfit_asscgd.sh
bash scripts/smoke_asscgd_adversarial.sh
bash scripts/smoke_asscgd_full_config.sh
bash scripts/train_asscgd_full.sh
```

The full script targets 25 epochs (47,450 slice updates), Adam at `2e-4`, an
adversarial weight of `0.1`, 100 diffusion steps, 10 inference steps, and BF16
generator convolutions. It writes one resumable checkpoint per epoch under
`/data/david_agyei/nyu-dataset/derived/asscgd_4x_stable` and automatically resumes the
latest checkpoint. Each training epoch selects a new deterministic mask partition;
the validation and test partitions remain fixed.

The original run's discriminator collapsed after epoch 2. Resume its last stable
generator while initializing the stabilized discriminator, and write the recovered
run to the separate stable directory:

```bash
ASSCGD_RESUME=/data/david_agyei/nyu-dataset/derived/asscgd_4x/checkpoint-epoch-002-step-0003796.pt \
ASSCGD_RESET_DISCRIMINATOR=1 \
bash scripts/train_asscgd_full.sh
```
