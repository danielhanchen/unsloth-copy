"""On a real Windows runner: does anything load sentencepiece?

Proving a negative, so this runs in two modes and the pair is the evidence:

  control    UNSLOTH_DISABLE_SENTENCEPIECE=0. sentencepiece MUST be loaded. If it is not,
             the probe cannot see loads at all and the treatment result means nothing.
  treatment  the shipped default. sentencepiece must NEVER be loaded.

Three independent detectors, because each can be fooled alone:

  1. sys.modules, the direct question.
  2. A meta-path watcher recording every find_spec for sentencepiece. With the sentinel in
     place an import short-circuits on sys.modules and never reaches the finders, so silence
     here is itself the proof; in the control it must be noisy.
  3. An audit hook on the "import" event, which records every import REQUEST, including the
     ones the sentinel refuses. That is not the same question as whether the extension
     loaded, and conflating the two reports a failure on a working fix; it is kept because
     naming who asked is the useful half.

Exits non-zero when the mode's expectation is not met.
"""

from __future__ import annotations

import importlib.abc
import json
import os
import sys
import traceback

MODE = sys.argv[1] if len(sys.argv) > 1 else "treatment"

FIND_SPEC_CALLS = []
AUDIT_IMPORTS = []


class Watcher(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path = None, target = None):
        if name == "sentencepiece" or name.startswith("sentencepiece."):
            FIND_SPEC_CALLS.append(
                {"name": name, "stack": [f.strip() for f in traceback.format_stack()[-6:-1]]}
            )
        return None      # never handle it, only observe


def _audit(event, args):
    if event != "import":
        return
    name = args[0] if args else ""
    if isinstance(name, str) and (name == "sentencepiece" or name.startswith("sentencepiece.")):
        AUDIT_IMPORTS.append(name)


sys.meta_path.insert(0, Watcher())
sys.addaudithook(_audit)

report = {
    "mode": MODE,
    "platform": sys.platform,
    "python": sys.version.split()[0],
    "env": os.environ.get("UNSLOTH_DISABLE_SENTENCEPIECE", "<unset>"),
    "steps": [],
    "errors": [],
}


def step(name, fn):
    try:
        report["steps"].append({name: fn()})
    except Exception as exc:
        report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")


def _sentencepiece_installed_on_disk():
    """Independent of sys.modules: is the package actually present in this environment?

    Without this the treatment leg passes trivially on a runner where sentencepiece was
    never installed, which is a green tick that proves nothing.
    """
    import importlib.metadata as md
    try:
        return md.version("sentencepiece")
    except Exception:
        return None


def _import_unsloth():
    """Import unsloth on a runner with no GPU.

    A hosted runner cannot finish this import. Without a spoof it raises NotImplementedError
    ("cannot find any torch accelerator"); with the repo's own tests/_zoo_aggressive_cuda_spoof
    it gets further and then hits the missing piece of the real stack instead, which is triton
    on windows-latest and an insufficient CUDA driver on ubuntu-latest.

    That does not make the leg meaningless, because the guard is the first thing
    unsloth/__init__.py runs, above the accelerator branch. So the import is allowed to fail,
    and the verdict below then requires the failure to be one of the known GPU-less-runner ones
    AND requires the guard's effect to be visible, which nothing else in this process produces.
    """
    import sys
    from pathlib import Path as _Path
    tests_dir = _Path(__file__).resolve().parents[1]
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    import _zoo_aggressive_cuda_spoof as _spoof
    _spoof.apply()
    import unsloth
    return getattr(unsloth, "__version__", "?")


# Every way a hosted runner is known to be unable to finish `import unsloth`. Anything else is
# a real failure and stays a VOID.
_NO_GPU_RUNNER_FAILURES = (
    "cannot find any torch accelerator",
    "No module named 'triton'",
    "CUDA driver version is insufficient",
)


def _import_transformers():
    import transformers
    return transformers.__version__


def _availability():
    from transformers.utils import import_utils
    return import_utils.is_sentencepiece_available()


def _build_tokenizers():
    from transformers import AutoTokenizer
    built = {}
    for model in (os.environ.get("PROBE_MODELS") or "").split(","):
        model = model.strip()
        if not model:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(model)
            built[model] = f"{type(tok).__name__} tokens={len(tok('hi there').input_ids)}"
        except Exception as exc:
            built[model] = f"FAIL {type(exc).__name__}: {str(exc)[:120]}"
    return built


step("sentencepiece_installed", _sentencepiece_installed_on_disk)
if os.environ.get("PROBE_IMPORT_UNSLOTH") == "1":
    step("import_unsloth", _import_unsloth)
else:
    # The Studio parent's rule, inlined the same way main.py does it, for the leg that
    # deliberately does not pay for the whole unsloth stack.
    value = (os.environ.get("UNSLOTH_DISABLE_SENTENCEPIECE") or "").strip().lower()
    if (
        value in ("1", "true", "yes", "on")
        or (value not in ("0", "false", "no", "off") and sys.platform == "win32")
    ) and "sentencepiece" not in sys.modules:
        sys.modules["sentencepiece"] = None
    report["steps"].append({"inlined_studio_rule": sys.modules.get("sentencepiece", "absent") is None})

step("import_transformers", _import_transformers)
step("is_sentencepiece_available", _availability)
step("tokenizers", _build_tokenizers)

alive = sorted(
    m for m, mod in sys.modules.items()
    if (m == "sentencepiece" or m.startswith("sentencepiece.")) and mod is not None
)
report["alive_sentencepiece_modules"] = alive
report["sentinel_present"] = sys.modules.get("sentencepiece", "absent") is None
report["find_spec_calls"] = len(FIND_SPEC_CALLS)
report["find_spec_detail"] = FIND_SPEC_CALLS[:3]
report["audit_imports"] = sorted(set(AUDIT_IMPORTS))

# An attempt is not a load. The audit hook fires when an import is requested, including the
# ones the sentinel then refuses with ImportError, and those are exactly the calls this change
# exists to intercept. Only a live module in sys.modules means the extension reached the
# loader, which on Windows is what raises the dialog.
loaded = bool(alive)
report["sentencepiece_was_loaded"] = loaded
report["import_attempts_refused"] = sorted(set(AUDIT_IMPORTS)) if not loaded else []

print(json.dumps(report, indent = 2))

installed = any("sentencepiece_installed" in s for s in report["steps"]) and next(
    (s["sentencepiece_installed"] for s in report["steps"] if "sentencepiece_installed" in s), None
)
if not installed:
    print("VERDICT: VOID, sentencepiece is not installed here, so nothing was being prevented")
    raise SystemExit(3)

if os.environ.get("PROBE_IMPORT_UNSLOTH") == "1" and not any(
    "import_unsloth" in s for s in report["steps"]
):
    failure = " ".join(report["errors"])
    reason = next((r for r in _NO_GPU_RUNNER_FAILURES if r in failure), None)
    if reason is None:
        # Nothing to do with the runner having no GPU, so unsloth genuinely did not import and
        # "unsloth did not load sentencepiece" would be true for the wrong reason. A green tick
        # here would be the emptiest kind.
        print("VERDICT: VOID, import unsloth failed:", report["errors"])
        raise SystemExit(5)
    if MODE == "treatment" and not report["sentinel_present"]:
        print("VERDICT: VOID, import unsloth stopped at", repr(reason),
              "before the guard ran, so this leg saw nothing:", report["errors"])
        raise SystemExit(5)
    # The guard is the first thing unsloth/__init__.py does and nothing else in this process
    # installs the sentinel, so its presence dates the failure to after the guard.
    report["import_unsloth_partial"] = reason
    print("NOTE: import unsloth could not finish on this runner", repr(reason) + ";",
          "the guard runs above that point and its effect is checked below")

if not any("import_transformers" in s for s in report["steps"]):
    print("VERDICT: VOID, transformers never imported:", report["errors"])
    raise SystemExit(6)

if MODE == "control":
    if not loaded:
        print("VERDICT: VOID, the control did not load sentencepiece, so this probe cannot "
              "detect a load and the treatment result is meaningless")
        raise SystemExit(4)
    print("VERDICT: control ok, sentencepiece loads when not disabled")
    raise SystemExit(0)

if loaded:
    print("VERDICT: FAIL, sentencepiece was loaded despite the Windows default")
    raise SystemExit(1)
print("VERDICT: PASS, sentencepiece was never loaded")
raise SystemExit(0)
