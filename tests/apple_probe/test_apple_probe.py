"""Apple-Silicon capability probe. Correctness/availability only.

GitHub-hosted macOS runners expose an MPS device whose allocations fail with a hard cap
(actions/runner-images#11899, #9918), so nothing here times anything. It answers:
what does Unsloth do on darwin/arm64, and is FlexAttention-on-MPS reachable at all.
"""
import platform, sys, json

REPORT = {}


def _record(key, fn):
    try:
        REPORT[key] = fn()
    except BaseException as e:            # a failure is a datapoint, not a test failure
        REPORT[key] = f"ERROR {type(e).__name__}: {e}"


def test_apple_probe():
    REPORT["platform"] = {"system": platform.system(), "machine": platform.machine(),
                          "python": sys.version.split()[0]}

    def torch_info():
        import torch
        d = {"version": torch.__version__,
             "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
             "mps_is_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
        # is_available() is not enough on CI runners -- probe a real allocation
        try:
            x = torch.zeros(8, 8, device="mps")
            d["mps_alloc_works"] = bool((x @ x).sum().item() == 0)
        except BaseException as e:
            d["mps_alloc_works"] = False
            d["mps_alloc_error"] = f"{type(e).__name__}: {e}"
        # How much MPS memory is actually usable? Reported failures on hosted runners
        # are at benchmark scale, not at 256 bytes, so find the real ceiling.
        ceiling = {}
        for mb in (1, 16, 128, 512, 1024, 2048):
            try:
                t = torch.empty(int(mb * 1024 * 1024 / 4), dtype=torch.float32, device="mps")
                t.fill_(1.0); torch.mps.synchronize()
                ceiling[f"{mb}MB"] = "ok"
                del t; torch.mps.empty_cache()
            except BaseException as e:
                ceiling[f"{mb}MB"] = f"{type(e).__name__}"
                break
        d["mps_alloc_ceiling"] = ceiling
        return d
    _record("torch", torch_info)

    def flex_info():
        import torch
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        d = {"importable": True}
        if not REPORT.get("torch", {}).get("mps_alloc_works"):
            d["mps_run"] = "skipped: MPS device unusable on this runner"
            return d
        def causal(b, h, q, k): return q >= k
        bm = create_block_mask(causal, B=None, H=None, Q_LEN=128, KV_LEN=128, device="mps")
        q = torch.randn(1, 2, 128, 32, device="mps")
        out = flex_attention(q, q, q, block_mask=bm)
        d["mps_run"] = f"ok shape={tuple(out.shape)}"
        return d
    _record("flex_attention", flex_info)

    def mlx_info():
        import mlx.core as mx
        a = mx.ones((4, 4))
        return {"importable": True, "version": getattr(__import__("mlx"), "__version__", "?"),
                "matmul_ok": float((a @ a).sum()) == 64.0,
                "default_device": str(mx.default_device())}
    _record("mlx", mlx_info)

    def unsloth_info():
        import unsloth
        from unsloth.device_type import get_device_type, DEVICE_TYPE, DEVICE_TYPE_TORCH
        return {"import_ok": True, "version": getattr(unsloth, "__version__", "?"),
                "get_device_type()": get_device_type(),
                "DEVICE_TYPE": DEVICE_TYPE, "DEVICE_TYPE_TORCH": DEVICE_TYPE_TORCH,
                "FastLanguageModel": type(unsloth.FastLanguageModel).__name__}
    _record("unsloth", unsloth_info)

    print("APPLE_PROBE_JSON_BEGIN")
    print(json.dumps(REPORT, indent=2))
    print("APPLE_PROBE_JSON_END")
    assert REPORT["platform"]["system"], "probe produced no platform info"
