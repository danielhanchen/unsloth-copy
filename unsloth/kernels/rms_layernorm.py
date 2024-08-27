import triton
import triton.language as tl
import torch
from .utils import calculate_settings


@triton.jit
def _rms_layernorm_forward(
    Y,
    Y_row_stride,
    X,
    X_row_stride,
    W,
    W_row_stride,
    r,
    r_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE : tl.constexpr
):
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)#.to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)

    tl.store(r, inv_var)

    normed = X_row * inv_var
    normed = normed.to(W_row.dtype) # Exact copy from HF
    
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)



@triton.jit
def _rms_layernorm_backward(
    dY,
    dY_row_stride,
    X,
    X_row_stride,
    W,
    W_row_stride,
    r,
    r_row_stride,
    dW,
    dW_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE : tl.constexpr,
):
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx *  r_row_stride

    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask=mask, other=0).to(tl.float32)

    # Get saved row variance
    inv_var = tl.load(r).to(tl.float32)

    dY_W = dY_row * W_row # < Liger added
    normed = X_row * inv_var

    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    output = inv_var/n_cols * (n_cols*dY_W - normed*rowsum_dY_normed)
    tl.store(dY + col_offsets, output, mask=mask)

    # calculate the gradient of W
    tl.store(dW_ptr + col_offsets, dY_normed, mask=mask) < Liger added
pass


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps, gemma = False):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device="cuda:0")
        r = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")

        _rms_layernorm_forward[(n_rows,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            W.stride(0),
            r,
            r.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps

        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)
    pass

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        dW = X

        _rms_layernorm_backward[(n_rows,)](
            dY,
            dY.stride(0),
            X,
            X .stride(0),
            W,
            W .stride(0),
            r,
            r .stride(0),
            dW,
            dW.stride(0),
            n_cols,
            ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        dX = dY.view(*shape)
        return dX, None, None, None
    pass
pass


def fast_rms_layernorm(layernorm, X, gemma = False):
    W   = layernorm.weight
    eps = layernorm.variance_epsilon
    out = Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    return out
pass
