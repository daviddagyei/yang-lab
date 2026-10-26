# External reference repositories

`mri-variationalnetwork/` is a local reference clone of the official VLOGroup
implementation and is intentionally ignored rather than vendored into this
project. The architecture adaptation in `mri_variational_network/` was checked
against upstream commit:

`653630b9dcd0687387e6cfa494fa3d049abb83d8`

Recreate the reference clone with:

```bash
git clone https://github.com/VLOGroup/mri-variationalnetwork.git \
  third_party/mri-variationalnetwork
git -C third_party/mri-variationalnetwork checkout \
  653630b9dcd0687387e6cfa494fa3d049abb83d8
```

`Deep-MRI-Reconstruction/` is the official reference implementation for
Schlemper et al.'s deep cascade CNN. The multicoil adaptation in
`mri_cascade_cnn/` was checked against upstream commit:

`d8a40efd892e57799c3413630e5cb92d5b035cf8`

Recreate it with:

```bash
git clone https://github.com/js3611/Deep-MRI-Reconstruction.git \
  third_party/Deep-MRI-Reconstruction
git -C third_party/Deep-MRI-Reconstruction checkout \
  d8a40efd892e57799c3413630e5cb92d5b035cf8
```

`AdaDiff/` optionally supplies the official fused StyleGAN `upfirdn2d` CUDA
source used to cross-check the production Triton backend. It is pinned to:

`15f9e1a04c13678d7c3c60ca4a47cf7b12482b24`

The directory is intentionally ignored by Git. Clone that commit when testing
`ADADIFF_FIR_BACKEND=official`; routine training uses the compiler-free Triton
backend bundled with the PyTorch CUDA installation.
