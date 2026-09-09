#!/usr/bin/env python3
"""Emit the GitHub Actions matrices for bisect-update.yml.

    python ci_plan.py --repo owner/name --oses a,b,c \
        [--pairs "806-807:staged,806-807:plain"] [--chain 0|1] [--policy 0|1] \
        [--pr-wheel 0|1] [--upgrade-from 805,806,807] [--jobs upgrade,faults]

Prints one `key=json` line per matrix for $GITHUB_OUTPUT, plus a `has_<key>` boolean for the job
`if:`. Release-bisection matrices: pairs, chain, policy. PR-verification matrices (only when
`--pr-wheel 1`): upgrade, oldshell, nested, faults, prefetch, timing, cargo.

Which PR jobs a repo runs comes from the `pr_matrix` section of ci_plan.json, so the three staging
repos share the matrix instead of each running all of it -- one PR's verification is a single pass
over the three repos, not three identical passes.
"""
import argparse
import json
import pathlib

PR_JOBS = ("upgrade", "oldshell", "nested", "faults", "prefetch", "timing", "cargo")
# `nested-old-stage` drives the 806 CLI's own POSIX staging path; there is no Windows twin.
POSIX_ONLY = ("nested",)

here = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--oses", default="ubuntu-latest,macos-15,windows-latest")
ap.add_argument("--pairs", default="")
ap.add_argument("--chain", default="")
ap.add_argument("--policy", default="")
ap.add_argument("--pr-wheel", default="")
ap.add_argument("--upgrade-from", default="", help="comma list of releases to upgrade from, e.g. 805,806,807")
ap.add_argument("--jobs", default="", help="comma list restricting which PR jobs run (empty = the repo's plan)")
a = ap.parse_args()

plan = json.loads((here / "ci_plan.json").read_text())
base = plan.get(a.repo, plan["_default"])
oses = [o for o in a.oses.split(",") if o]
pr_wheel = a.pr_wheel.strip().lower() in ("1", "true", "yes")

# ---------------------------------------------------------------- release bisection matrices
pairs = list(base["pairs"])
if a.pairs.strip():
    pairs = []
    for item in a.pairs.split(","):
        item = item.strip()
        if not item:
            continue
        rng, _, mode = item.partition(":")
        f, t = rng.split("-")
        pairs.append([int(f), int(t), mode or "plain"])
elif pr_wheel:
    # A PR run is about the PR, not about re-timing the release history.
    pairs = []
chain = base["chain"] if a.chain == "" else a.chain.lower() in ("1", "true", "yes")
policy = base["policy"] if a.policy == "" else a.policy.lower() in ("1", "true", "yes")
if pr_wheel and a.chain == "":
    chain = False
if pr_wheel and a.policy == "":
    policy = base.get("pr_matrix", {}).get("policy", False)

pair_inc = [{"os": o, "from": f, "to": t, "mode": m} for o in oses for f, t, m in pairs]
chain_inc = [{"os": o} for o in oses] if chain else []
policy_inc = [{"os": o} for o in oses] if policy else []

# ---------------------------------------------------------------- PR verification matrices
pr = dict(plan["_default"].get("pr_matrix", {}))
pr.update(base.get("pr_matrix", {}))
wanted = [j for j in (a.jobs.split(",") if a.jobs.strip() else PR_JOBS) if j in PR_JOBS]

upgrade_from = [x.strip() for x in a.upgrade_from.split(",") if x.strip()] or [str(x) for x in pr.get("upgrade_from", [])]
inc = {j: [] for j in PR_JOBS}
if pr_wheel:
    for job in wanted:
        job_oses = [o for o in oses if not (job in POSIX_ONLY and "windows" in o)]
        if job == "upgrade":
            inc[job] = [{"os": o, "from": n} for o in job_oses for n in upgrade_from]
        elif pr.get(job):
            inc[job] = [{"os": o} for o in job_oses]

out = {"pairs": {"include": pair_inc}, "chain": {"include": chain_inc}, "policy": {"include": policy_inc}}
out.update({j: {"include": inc[j]} for j in PR_JOBS})
for key, value in out.items():
    print(f"{key}=" + json.dumps(value))
    print(f"has_{key}=" + ("true" if value["include"] else "false"))
