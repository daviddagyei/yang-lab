# Multicoil MoDL

This package adapts the recursive MoDL architecture to the project's PyTorch
multicoil pipeline. It alternates a shared residual CNN with a differentiable
complex conjugate-gradient solution of

```text
(A^H A + lambda I)x = A^H y + lambda D_w(x).
```

`A` and `A^H` come from `mri_reconstruction`, so MoDL uses the same supplied
sensitivity maps, Cartesian masks, FFT convention, cropping, and normalization as
the RDBM experiment.

Primary reference: H. K. Aggarwal, M. P. Mani, and M. Jacob, *MoDL: Model-Based
Deep Learning Architecture for Inverse Problems*, IEEE TMI 38(2), 394-405,
https://doi.org/10.1109/TMI.2018.2865356.

Authors' implementation: https://github.com/hkaggarwal/modl
