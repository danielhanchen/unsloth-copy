# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Warm-versus-cold token parity for Studio's MLX VLM prompt-prefix cache.

The mechanism's promise is exactness: a turn served from a stored snapshot must produce
the same tokens as the same turn prefilled from scratch. Nothing in CI checks that --
the unit suite fakes ``mlx.core`` and never imports ``mlx_vlm``, and the Apple Silicon
job only proves imports and a text-only training smoke. This runs the real thing.

Two series over one fixed conversation:

  warm  every turn through the store, as a chat does
  cold  the same turns with the store cleared between them

Both series prefill on the same 256-token grid, so the only difference between them is
whether rows were reused. "Cold" is NOT the store disabled: disabling it also drops the
grid, which is a different prefill policy and cannot serve as the reference.

The turns are scripted, identical for both series. Feeding a series its own answers
would let one divergence rewrite every later prompt and hide where it started.

Usage:
    python tests/studio/run_real_mlx_vlm_cache_smoke.py [--model REPO] [--revision SHA]
        [--turns N] [--max-new-tokens N] [--json PATH] [--allow-no-reuse]
"""

import argparse
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "studio" / "backend"))

DEFAULT_MODEL = "mlx-community/SmolVLM-256M-Instruct-4bit"

# Long enough that turn 1 already crosses a 256-row boundary, so every later turn has a
# stored prefix to resume. A short chat has no boundary at all and would reuse nothing,
# which is by design and would make this script pass without testing anything.
PREAMBLE = (
    "You are a careful assistant. Answer in one short sentence. "
    + "Keep the following reference notes in mind while you answer. "
    + " ".join(
        f"Note {n}: the value of parameter p{n} is {n * 7 % 97} and it is used by stage {n % 5}."
        for n in range(1, 60)
    )
)

QUESTIONS = [
    "What is the value of parameter p3?",
    "Which stage uses parameter p8?",
    "Name the parameter whose value is largest among p1, p2 and p3.",
    "How many notes mention stage 0?",
    "Summarise the notes in one sentence.",
]

ANSWERS = [
    "Parameter p3 is 21.",
    "Stage 3 uses parameter p8.",
    "p3 has the largest value of the three.",
    "Several notes mention stage 0.",
    "The notes list parameter values and the stages that use them.",
]


def log(message):
    print(message, flush = True)


def preflight():
    """Refuse to report a pass from a host or a runtime that cannot prove anything."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit(f"needs Apple Silicon, got {platform.system()}/{platform.machine()}")
    import mlx.core as mx
    if getattr(mx, "__file__", None) is None:
        raise SystemExit("mlx resolved to a stub, not the real package")
    value = mx.sum(mx.arange(1024, dtype = mx.float32))
    mx.eval(value)
    assert float(value.item()) == 523776.0, "real mlx did not compute"
    import mlx_vlm
    versions = {
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "mlx": getattr(mx, "__version__", "?"),
        "mlx_vlm": getattr(mlx_vlm, "__version__", "?"),
        "metal": bool(getattr(getattr(mx, "metal", None), "is_available", lambda: False)()),
    }
    log(f"[preflight] {json.dumps(versions)}")
    return versions


def make_image():
    """A deterministic image, so the digest that keys the store is reproducible."""
    from PIL import Image
    image = Image.new("RGB", (64, 64), (30, 60, 90))
    for x in range(64):
        for y in range(64):
            image.putpixel((x, y), ((x * 4) % 256, (y * 4) % 256, ((x + y) * 2) % 256))
    return image


class Observer:
    """Collects the token ids mlx-vlm actually emitted, and the kwargs it was given.

    Re-encoding the rendered answer would not do: two different token sequences can
    decode to the same string, which is exactly the divergence worth catching.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.token_ids = []
        self.prefill_step = None
        self.had_session = None
        self.final = None

    def wrap(self, real_stream):
        def traced(*args, **kwargs):
            self.prefill_step = kwargs.get("prefill_step_size")
            self.had_session = kwargs.get("prompt_cache_state") is not None
            emitted = 0
            for event in real_stream(*args, **kwargs):
                count = int(getattr(event, "generation_tokens", 0) or 0)
                token = getattr(event, "token", None)
                # The terminal event repeats the last token; the counter, not the token
                # value, says whether it is new. Identical consecutive tokens are valid.
                if count > emitted and token is not None:
                    self.token_ids.append(int(token))
                    emitted = count
                self.final = event
                yield event
        return traced


def run_turn(backend, observer, messages, image, max_new_tokens):
    import mlx_vlm

    observer.reset()
    # ``_generate_vlm`` does `from mlx_vlm import stream_generate` at call time, so
    # rebinding the attribute here is what the request will pick up.
    real = mlx_vlm.stream_generate
    mlx_vlm.stream_generate = observer.wrap(real)
    started = time.perf_counter()
    try:
        text = ""
        for snapshot in backend.generate_chat_response(
            copy.deepcopy(messages),
            image = image,
            temperature = 0.0,
            top_p = 1.0,
            top_k = 0,
            min_p = 0.0,
            max_new_tokens = max_new_tokens,
            repetition_penalty = 1.0,
            presence_penalty = 0.0,
            frequency_penalty = 0.0,
            seed = 3407,
        ):
            text = snapshot          # cumulative snapshots, not deltas
    finally:
        mlx_vlm.stream_generate = real
    elapsed = time.perf_counter() - started

    stats = copy.deepcopy(backend.last_generation_stats or {})
    usage = stats.get("usage", {})
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
    return {
        "ids": tuple(observer.token_ids),
        "text": text,
        "cached_tokens": cached,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "prefill_step": observer.prefill_step,
        "had_session": observer.had_session,
        "seconds": round(elapsed, 3),
    }


def conversation(turns):
    """Turn i's messages: the scripted history, then question i."""
    for index in range(turns):
        messages = []
        for earlier in range(index):
            messages.append({
                "role": "user",
                "content": (PREAMBLE + " " + QUESTIONS[earlier]) if earlier == 0
                           else QUESTIONS[earlier],
            })
            messages.append({"role": "assistant", "content": ANSWERS[earlier]})
        messages.append({
            "role": "user",
            "content": (PREAMBLE + " " + QUESTIONS[index]) if index == 0 else QUESTIONS[index],
        })
        yield messages


def load_backend(model, max_seq_length = 8192):
    from core.inference.mlx_inference import MLXInferenceBackend
    backend = MLXInferenceBackend()
    ok = backend.load_model(
        SimpleNamespace(identifier = model, is_vision = True, is_lora = False, is_gguf = False),
        max_seq_length = max_seq_length,
    )
    if not ok:
        raise SystemExit(f"could not load {model}")
    return backend


def series(backend, observer, turns, image, max_new_tokens, cold):
    results = []
    for messages in conversation(turns):
        if cold:
            backend._clear_prompt_cache()
        results.append(run_turn(backend, observer, messages, image, max_new_tokens))
    return results


def compare(name, warm, cold, allow_no_reuse):
    """Every warm turn must match its cold twin exactly, and reuse must have happened."""
    problems = []
    for index, (w, c) in enumerate(zip(warm, cold), start = 1):
        log(
            f"  turn {index}: warm cached={w['cached_tokens']:>5} prompt={w['prompt_tokens']:>5} "
            f"{w['seconds']:>6}s | cold cached={c['cached_tokens']:>5} "
            f"prompt={c['prompt_tokens']:>5} {c['seconds']:>6}s | "
            f"{'MATCH' if w['ids'] == c['ids'] else 'DIVERGED'}"
        )
        if w["ids"] != c["ids"]:
            first = next(
                (i for i, (a, b) in enumerate(zip(w["ids"], c["ids"])) if a != b),
                min(len(w["ids"]), len(c["ids"])),
            )
            problems.append(
                f"{name} turn {index}: tokens diverge at position {first}; "
                f"warm={w['ids'][first:first + 8]} cold={c['ids'][first:first + 8]}"
            )
        if c["cached_tokens"]:
            problems.append(f"{name} turn {index}: the cold reference reported reuse")
        if w["prefill_step"] != c["prefill_step"]:
            problems.append(
                f"{name} turn {index}: prefill step differs between the series "
                f"({w['prefill_step']} vs {c['prefill_step']}), so they are not comparable"
            )
    reused = sum(1 for w in warm[1:] if w["cached_tokens"] > 0)
    log(f"  {name}: {reused}/{max(len(warm) - 1, 0)} later warm turns reused a prefix")
    if not reused and not allow_no_reuse:
        problems.append(
            f"{name}: no warm turn reused anything, so token equality proves nothing"
        )
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = DEFAULT_MODEL)
    parser.add_argument("--revision", default = None)
    parser.add_argument("--turns", type = int, default = 5)
    parser.add_argument("--max-new-tokens", type = int, default = 24)
    parser.add_argument("--json", default = None)
    parser.add_argument(
        "--allow-no-reuse", action = "store_true",
        help = "report, rather than fail, when no turn reused a prefix (image chats on "
               "causal models only reach a boundary after a few turns)",
    )
    parser.add_argument("--skip-image", action = "store_true")
    args = parser.parse_args()

    versions = preflight()
    if args.revision:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    log(f"[load] {args.model}")
    backend = load_backend(args.model)
    observer = Observer()
    report = {"model": args.model, "versions": versions, "lanes": {}}
    problems = []

    log("[text] warm series")
    warm = series(backend, observer, args.turns, None, args.max_new_tokens, cold = False)
    log("[text] cold series")
    backend._clear_prompt_cache()
    cold = series(backend, observer, args.turns, None, args.max_new_tokens, cold = True)
    problems += compare("text", warm, cold, allow_no_reuse = False)
    report["lanes"]["text"] = {"warm": warm, "cold": cold}

    if not any(turn["had_session"] for turn in warm):
        problems.append(
            "no request built a prompt-cache session at all: the store disabled itself, "
            "so this run tested the unchanged path"
        )

    if not args.skip_image:
        image = make_image()
        log("[image] warm series")
        backend._clear_prompt_cache()
        warm_i = series(backend, observer, args.turns, image, args.max_new_tokens, cold = False)
        log("[image] cold series")
        backend._clear_prompt_cache()
        cold_i = series(backend, observer, args.turns, image, args.max_new_tokens, cold = True)
        # An image chat on a causal model only reaches a boundary past the image after a
        # few turns, so reuse is reported rather than required.
        problems += compare("image", warm_i, cold_i, allow_no_reuse = args.allow_no_reuse)
        report["lanes"]["image"] = {"warm": warm_i, "cold": cold_i}

    report["problems"] = problems
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent = 2), encoding = "utf-8")

    if problems:
        log("\nFAILURES:")
        for problem in problems:
            log(f"  - {problem}")
        return 1
    log("\nwarm and cold agree token for token, and reuse happened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
