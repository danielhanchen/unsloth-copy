"""On a real Windows runner: does a Studio worker load sentencepiece?

Studio's workers are spawned interpreters, so they inherit nothing from the parent's
sys.modules, and each of them imports transformers long before it imports unsloth. They all
start in one place, utils.native_path_leases.run_without_native_path_secret, which resolves
the worker module by name and imports it. This drives that real entrypoint with a stand-in
worker module that does what a worker does next: import transformers and build a tokenizer.

Two legs, and the pair is the evidence, exactly as in ci_probe_sentencepiece.py:

  control    UNSLOTH_DISABLE_SENTENCEPIECE=0. sentencepiece MUST be loaded, or this probe
             cannot detect a load and the treatment result means nothing.
  treatment  the shipped default. sentencepiece must NEVER be loaded, and the sentinel must
             already be in place at the moment the worker module is imported, which is the
             property under test: installed afterwards it would leave transformers reporting
             the package available while importing it fails.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "treatment"

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio" / "backend"

WORKER = textwrap.dedent(
    '''
    """Stands in for core.training.worker: same position, same first moves."""
    import os, sys

    # Module scope, which is the moment the entrypoint imports us. A sentinel installed after
    # this point is too late for everything below.
    AT_IMPORT = sys.modules.get("sentencepiece", "absent") is None
    ENV_APPLIED = os.environ.get("UNSLOTH_STUDIO_SP_PROBE")

    def report():
        import transformers
        from transformers.utils import import_utils
        result = {
            "sentinel_at_worker_import": AT_IMPORT,
            "env_applied": ENV_APPLIED,
            "transformers": transformers.__version__,
            "is_sentencepiece_available": import_utils.is_sentencepiece_available(),
            "tokenizers": {},
        }
        from transformers import AutoTokenizer
        for model in (os.environ.get("PROBE_MODELS") or "").split(","):
            model = model.strip()
            if not model:
                continue
            try:
                tok = AutoTokenizer.from_pretrained(model)
                result["tokenizers"][model] = (
                    f"{type(tok).__name__} tokens={len(tok('hi there').input_ids)}"
                )
            except Exception as exc:
                result["tokenizers"][model] = f"FAIL {type(exc).__name__}: {str(exc)[:120]}"
        result["alive_sentencepiece_modules"] = sorted(
            m for m, mod in sys.modules.items()
            if (m == "sentencepiece" or m.startswith("sentencepiece.")) and mod is not None
        )
        return result
    '''
)


def _installed_on_disk():
    """A runner without sentencepiece would pass the treatment leg for the wrong reason."""
    import importlib.metadata as md
    try:
        return md.version("sentencepiece")
    except Exception:
        return None


installed = _installed_on_disk()
work = Path(tempfile.mkdtemp(prefix = "spwin_worker_probe_"))
(work / "spwin_worker_probe.py").write_text(WORKER, encoding = "utf-8")
sys.path.insert(0, str(work))
sys.path.insert(0, str(BACKEND))

from utils.native_path_leases import run_without_native_path_secret  # noqa: E402

assert "sentencepiece" not in sys.modules, sys.modules["sentencepiece"]
result = run_without_native_path_secret(
    "spwin_worker_probe", "report", {"UNSLOTH_STUDIO_SP_PROBE": "yes"}
)

report = {
    "mode": MODE,
    "platform": sys.platform,
    "python": sys.version.split()[0],
    "env": os.environ.get("UNSLOTH_DISABLE_SENTENCEPIECE", "<unset>"),
    "sentencepiece_installed": installed,
    **result,
}
loaded = bool(report["alive_sentencepiece_modules"])
report["sentencepiece_was_loaded"] = loaded
print(json.dumps(report, indent = 2))

if not installed:
    print("VERDICT: VOID, sentencepiece is not installed here, so nothing was being prevented")
    raise SystemExit(3)
if report["env_applied"] != "yes":
    print("VERDICT: VOID, the entrypoint did not run, so nothing here was exercised")
    raise SystemExit(6)

if MODE == "control":
    if not loaded:
        print("VERDICT: VOID, the control did not load sentencepiece, so this probe cannot "
              "detect a load and the treatment result is meaningless")
        raise SystemExit(4)
    print("VERDICT: control ok, a worker loads sentencepiece when not disabled")
    raise SystemExit(0)

if not report["sentinel_at_worker_import"]:
    print("VERDICT: FAIL, the worker module was imported before the sentinel was installed")
    raise SystemExit(2)
if loaded:
    print("VERDICT: FAIL, sentencepiece was loaded in the worker despite the Windows default")
    raise SystemExit(1)
print("VERDICT: PASS, the worker never loaded sentencepiece and the sentinel was in place "
      "before the worker module was imported")
raise SystemExit(0)
