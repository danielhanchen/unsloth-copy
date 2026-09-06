"""MLX vs PyTorch FlexAttention vs PyTorch SDPA, on real Apple Silicon.

FORWARD ONLY. torch 2.14 flex_attention.py:2214 raises NotImplementedError for backward
on MPS, so this cannot speak to training -- only to inference/prefill.

Patterns mirror what Unsloth builds: full causal and a sliding-window band. The SDPA
arm passes a DENSE (T,T) -inf mask, which is what unsloth/utils/packing.py and
attention_dispatch.py actually construct.
"""
import json, time, platform

REPORT = {"platform": {"system": platform.system(), "machine": platform.machine()}}


def _bench(fn, sync, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def test_attention_bench():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    import mlx.core as mx

    REPORT["torch_version"] = torch.__version__
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.accumulated_recompile_limit = 512
    torch.manual_seed(0)
    H, D, WINDOW = 8, 64, 512
    sync_t = torch.mps.synchronize
    rows = []

    for T in (1024, 2048, 4096):
        for pattern in ("causal", "sliding_window"):
            row = {"T": T, "pattern": pattern}
            try:
                q = torch.randn(1, H, T, D, device="mps", dtype=torch.float16)
                k = torch.randn(1, H, T, D, device="mps", dtype=torch.float16)
                v = torch.randn(1, H, T, D, device="mps", dtype=torch.float16)

                i = torch.arange(T, device="mps")
                dist = i[:, None] - i[None, :]
                ok = (dist >= 0) if pattern == "causal" else ((dist >= 0) & (dist < WINDOW))
                dense = torch.where(ok, 0.0, float("-inf")).to(torch.float16)
                row["dense_mask_mb"] = dense.numel() * dense.element_size() / 1024**2

                row["sdpa_dense_ms"] = _bench(
                    lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=dense), sync_t)

                if pattern == "causal":
                    def mm(b, h, qi, ki): return qi >= ki
                else:
                    def mm(b, h, qi, ki): return (qi >= ki) & (qi - ki < WINDOW)
                bm = create_block_mask(mm, B=None, H=None, Q_LEN=T, KV_LEN=T, device="mps")
                row["block_sparsity_pct"] = float(bm.sparsity())
                # MUST be compiled. Bare flex_attention() falls back to an unfused path
                # that materialises the full score matrix -- which measures the opposite
                # of what FlexAttention is for.
                flex_c = torch.compile(flex_attention, dynamic=False)
                flex_c(q, k, v, block_mask=bm)          # compile before timing
                sync_t()
                row["flex_ms"] = _bench(lambda: flex_c(q, k, v, block_mask=bm), sync_t)

                # MLX: fp16, same shapes. mlx uses [B, H, T, D] too.
                mq = mx.random.normal((1, H, T, D)).astype(mx.float16)
                mk = mx.random.normal((1, H, T, D)).astype(mx.float16)
                mv = mx.random.normal((1, H, T, D)).astype(mx.float16)
                scale = 1.0 / (D ** 0.5)
                if pattern == "causal":
                    mmask = "causal"
                else:
                    mi = mx.arange(T)
                    md = mi[:, None] - mi[None, :]
                    mmask = mx.where((md >= 0) & (md < WINDOW), mx.array(0.0, mx.float16),
                                     mx.array(-6e4, mx.float16))
                def mlx_run():
                    o = mx.fast.scaled_dot_product_attention(mq, mk, mv, scale=scale, mask=mmask)
                    mx.eval(o)
                row["mlx_ms"] = _bench(mlx_run, mx.synchronize)

                row["flex_vs_sdpa"] = row["sdpa_dense_ms"] / row["flex_ms"]
                row["mlx_vs_flex"] = row["flex_ms"] / row["mlx_ms"]
                del q, k, v, dense, bm
                torch.mps.empty_cache()
            except BaseException as e:
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)

    REPORT["rows"] = rows
    print("ATTN_BENCH_JSON_BEGIN")
    print(json.dumps(REPORT, indent=2))
    print("ATTN_BENCH_JSON_END")
    assert rows
