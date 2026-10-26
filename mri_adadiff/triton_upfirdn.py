"""Fused Triton upsample/filter/downsample with double-backward support.

This implements the same operator and autograd structure as the fused
StyleGAN/AdaDiff CUDA path without requiring a system CUDA compiler.  The
kernel is intentionally limited to square FIR kernels, which covers every
AdaDiff resampling operation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _upfirdn2d_kernel(
    input_pointer,
    kernel_pointer,
    output_pointer,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    number_of_outputs: tl.constexpr,
    up: tl.constexpr,
    down: tl.constexpr,
    pad_before: tl.constexpr,
    kernel_size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    plane_size = output_height * output_width
    plane = offsets // plane_size
    within_plane = offsets - plane * plane_size
    output_y = within_plane // output_width
    output_x = within_plane - output_y * output_width
    valid_output = offsets < number_of_outputs
    accumulator = tl.zeros((block_size,), dtype=tl.float32)
    for kernel_y in tl.static_range(0, kernel_size):
        inserted_y = output_y * down + kernel_y - pad_before
        y_divisible = inserted_y % up == 0
        input_y = inserted_y // up
        valid_y = y_divisible & (input_y >= 0) & (input_y < input_height)
        for kernel_x in tl.static_range(0, kernel_size):
            inserted_x = output_x * down + kernel_x - pad_before
            x_divisible = inserted_x % up == 0
            input_x = inserted_x // up
            valid_x = x_divisible & (input_x >= 0) & (input_x < input_width)
            input_offset = (
                plane * input_height * input_width + input_y * input_width + input_x
            )
            # PyTorch conv2d is correlation; the reference path flips the FIR.
            kernel_offset = (
                (kernel_size - 1 - kernel_y) * kernel_size
                + (kernel_size - 1 - kernel_x)
            )
            sample = tl.load(
                input_pointer + input_offset,
                mask=valid_output & valid_y & valid_x,
                other=0.0,
            )
            coefficient = tl.load(kernel_pointer + kernel_offset)
            accumulator += sample.to(tl.float32) * coefficient.to(tl.float32)
    tl.store(output_pointer + offsets, accumulator, mask=valid_output)


def _launch(
    value: torch.Tensor,
    kernel: torch.Tensor,
    *,
    up: int,
    down: int,
    pad: tuple[int, int],
) -> torch.Tensor:
    if not value.is_cuda or not kernel.is_cuda:
        raise ValueError("Triton upfirdn2d requires CUDA tensors")
    if value.ndim != 4 or kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("invalid Triton upfirdn2d shapes")
    batch, channels, input_height, input_width = value.shape
    pad_before, pad_after = pad
    kernel_size = kernel.shape[0]
    output_height = (input_height * up + pad_before + pad_after - kernel_size) // down + 1
    output_width = (input_width * up + pad_before + pad_after - kernel_size) // down + 1
    output = torch.empty(
        (batch, channels, output_height, output_width),
        device=value.device,
        dtype=value.dtype,
    )
    number_of_outputs = output.numel()
    block_size = 256
    grid = (triton.cdiv(number_of_outputs, block_size),)
    _upfirdn2d_kernel[grid](
        value.contiguous(),
        kernel.contiguous(),
        output,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        number_of_outputs=number_of_outputs,
        up=up,
        down=down,
        pad_before=pad_before,
        kernel_size=kernel_size,
        block_size=block_size,
    )
    return output


class _UpFirDnBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gradient_output: torch.Tensor,
        kernel: torch.Tensor,
        gradient_kernel: torch.Tensor,
        up: int,
        down: int,
        pad: tuple[int, int],
        gradient_pad: tuple[int, int],
        input_size: torch.Size,
    ) -> torch.Tensor:
        gradient_input = _launch(
            gradient_output,
            gradient_kernel,
            up=down,
            down=up,
            pad=gradient_pad,
        )
        if gradient_input.shape != input_size:
            raise RuntimeError(
                f"Triton FIR backward shape {gradient_input.shape} != {input_size}"
            )
        ctx.save_for_backward(kernel)
        ctx.up = up
        ctx.down = down
        ctx.pad = pad
        return gradient_input

    @staticmethod
    def backward(ctx, gradient_gradient_input: torch.Tensor):
        (kernel,) = ctx.saved_tensors
        gradient_gradient_output = _launch(
            gradient_gradient_input,
            kernel,
            up=ctx.up,
            down=ctx.down,
            pad=ctx.pad,
        )
        return gradient_gradient_output, None, None, None, None, None, None, None


class _UpFirDn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        value: torch.Tensor,
        kernel: torch.Tensor,
        up: int,
        down: int,
        pad: tuple[int, int],
    ) -> torch.Tensor:
        output = _launch(value, kernel, up=up, down=down, pad=pad)
        input_height, input_width = value.shape[-2:]
        output_height, output_width = output.shape[-2:]
        pad_before, _ = pad
        kernel_size = kernel.shape[0]
        gradient_pad_before = kernel_size - pad_before - 1
        gradient_pad_after_height = (
            input_height * up - output_height * down + pad_before - up + 1
        )
        gradient_pad_after_width = (
            input_width * up - output_width * down + pad_before - up + 1
        )
        if gradient_pad_after_height != gradient_pad_after_width:
            raise RuntimeError("AdaDiff Triton FIR expects square spatial dimensions")
        ctx.save_for_backward(kernel, torch.flip(kernel, (0, 1)))
        ctx.up = up
        ctx.down = down
        ctx.pad = pad
        ctx.gradient_pad = (gradient_pad_before, gradient_pad_after_height)
        ctx.input_size = value.shape
        return output

    @staticmethod
    def backward(ctx, gradient_output: torch.Tensor):
        kernel, gradient_kernel = ctx.saved_tensors
        gradient_input = _UpFirDnBackward.apply(
            gradient_output,
            kernel,
            gradient_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.gradient_pad,
            ctx.input_size,
        )
        return gradient_input, None, None, None, None


def triton_upfirdn2d(
    value: torch.Tensor,
    kernel: torch.Tensor,
    *,
    up: int = 1,
    down: int = 1,
    pad: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Apply fused FIR resampling with first- and second-order autograd."""

    if not value.is_cuda:
        raise ValueError("Triton FIR backend is CUDA-only")
    return _UpFirDn.apply(value, kernel, int(up), int(down), tuple(pad))
