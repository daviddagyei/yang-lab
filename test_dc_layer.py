"""
Local test of the data consistency (DC) layer on a real brain image.

What it does:
  1. Loads one of your real HQ brain PNGs.
  2. Simulates the scanner skipping k-space lines (4x undersampling, same
     style of mask as preprocessing) to make the degraded input ("mu").
  3. Makes a stand-in "network prediction" (the true image plus noise).
     NOTE: this is NOT a trained model, just an imperfect guess so you can
     see what the DC layer does to it.
  4. Runs the DC layer from rdbm.py on that guess.
  5. Saves ONE figure + prints numbers you can send to Moj.

Usage (from the folder that holds this file and rdbm.py):
    python test_dc_layer.py --hq_image /path/to/one_hq_image.png

Requires: torch, numpy, Pillow, matplotlib
    pip install torch numpy Pillow matplotlib
(If importing rdbm.py also needs einops etc., the script falls back to a
built-in copy of the same function and tells you it did.)
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1))


def ifft2c(k):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))), dim=(-2, -1))


def fallback_dc(x_start, mu, mask, dc_weight=1.0):
    """Same math as RDBM.data_consistency, used only if rdbm.py can't be imported."""
    xk, mk = fft2c(x_start), fft2c(mu)
    return ifft2c(xk * (1 - mask) + (dc_weight * mk + (1 - dc_weight) * xk) * mask).real


def get_dc_function():
    """Prefer the real method from rdbm.py so we are testing YOUR code."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    try:
        from rdbm import RDBM
        print("Using data_consistency from rdbm.py (your actual code).")
        return lambda x, mu, mask: RDBM.data_consistency(None, x, mu, mask), "rdbm.py"
    except Exception as e:
        print(f"Could not import rdbm.py ({type(e).__name__}: {e}).")
        print("Using a built-in copy of the same function instead.")
        return lambda x, mu, mask: fallback_dc(x, mu, mask), "built-in copy"


def make_mask(h, w, accel=4, center_frac=0.08):
    """Keep a fully sampled center strip plus every `accel`-th column."""
    cols = np.zeros(w, dtype=np.float32)
    n_center = int(round(w * center_frac))
    pad = (w - n_center + 1) // 2
    cols[pad:pad + n_center] = 1.0
    cols[::accel] = 1.0
    return torch.from_numpy(np.tile(cols[None, :], (h, 1)))  # (H, W)


def psnr(a, b):
    # images are in [-1, 1], so the data range is 2
    mse = torch.mean((a - b) ** 2).item()
    return 10 * np.log10(4.0 / max(mse, 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hq_image", required=True, help="Path to one real HQ brain PNG")
    p.add_argument("--output", default="dc_test_result.png")
    p.add_argument("--accel", type=int, default=4)
    p.add_argument("--noise", type=float, default=0.15,
                   help="Noise added to make the stand-in 'prediction' imperfect")
    args = p.parse_args()

    torch.manual_seed(0)
    dc_fn, source = get_dc_function()

    # --- load real image, grayscale, scale to [-1, 1] like RDBM's sample() does ---
    img = np.array(Image.open(args.hq_image).convert("L"), dtype=np.float32) / 255.0
    hq = torch.from_numpy(img * 2 - 1)[None, None]  # (1, 1, H, W)
    h, w = hq.shape[-2:]

    # --- simulate undersampled scan ---
    mask = make_mask(h, w, accel=args.accel)
    mu = ifft2c(fft2c(hq) * mask).real  # degraded input

    # --- stand-in imperfect "network prediction" ---
    pred = hq + args.noise * torch.randn_like(hq)

    # --- apply the DC layer ---
    corrected = dc_fn(pred, mu, mask)

    # --- numbers ---
    m = mask.bool()
    def k_err(a):
        d = (fft2c(a) - fft2c(mu))[..., m]
        return (d.abs().mean() / fft2c(mu)[..., m].abs().mean()).item()

    err_before, err_after = k_err(pred), k_err(corrected)
    psnr_mu, psnr_pred, psnr_corr = psnr(mu, hq), psnr(pred, hq), psnr(corrected, hq)

    print()
    print(f"Image size: {h}x{w}   Undersampling: {args.accel}x   "
          f"Fraction of k-space columns measured: {mask[0].mean().item():.2f}")
    print(f"K-space error at MEASURED locations (relative): "
          f"before DC = {err_before:.4f}, after DC = {err_after:.6f}")
    print(f"PSNR vs true image:  degraded input = {psnr_mu:.2f} dB | "
          f"stand-in prediction = {psnr_pred:.2f} dB | after DC = {psnr_corr:.2f} dB")

    # --- figure ---
    fig, ax = plt.subplots(1, 4, figsize=(14, 4.2))
    panels = [
        (hq, "True image (HQ)"),
        (mu, f"Degraded input ({args.accel}x)\nPSNR {psnr_mu:.1f} dB"),
        (pred, f"Stand-in prediction\n(NOT a trained model)\nPSNR {psnr_pred:.1f} dB"),
        (corrected, f"After data consistency\nPSNR {psnr_corr:.1f} dB"),
    ]
    for a, (t, title) in zip(ax, panels):
        a.imshow(t[0, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        a.set_title(title, fontsize=10)
        a.axis("off")
    fig.suptitle(
        f"Data consistency layer test ({source}) | k-space error at measured lines: "
        f"{err_before:.3f} -> {err_after:.1e}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved figure to: {args.output}")
    print("Send this PNG (and the numbers printed above) to Moj.")


if __name__ == "__main__":
    main()
