"""Read-only GitHub metadata intake. Never fetch source, diffs, logs or comments.

Run on a trusted host, not from a candidate checkout. Only allowlisted enums, validated
commit IDs, bounded counts, PR numbers and timestamps may leave this process.
Observed green CI is not proof that strict original-source security audits passed.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = "unslothai/unsloth"
REPO_ID = 725205304
ROOT = "/repos/" + REPO
API = "https://api.github.com"
LIMIT = 20
MAX_PAGES = 30
MAX_BYTES = 24 * 1024 * 1024
CONCLUSIONS = {"success", "failure", "neutral", "cancelled", "skipped", "timed_out", "action_required", "stale", "startup_failure"}
STATUSES = {"queued", "in_progress", "completed", "waiting", "requested", "pending"}
ERROR_CODES = {"HTTP_401", "HTTP_403", "HTTP_404", "HTTP_429", "HTTP_ERROR", "NETWORK_UNAVAILABLE", "INVALID_METADATA", "INCOMPLETE_PAGINATION", "RESPONSE_TOO_LARGE", "LIMIT_REACHED"}

class Blocked(Exception):
    def __init__(self, code):
        self.code = code if code in ERROR_CODES else "INVALID_METADATA"
        super().__init__(self.code)


def integer(value, minimum=0, maximum=10**15):
    if type(value) is not int or not minimum <= value <= maximum:
        raise Blocked("INVALID_METADATA")
    return value


def sha(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise Blocked("INVALID_METADATA")
    return value


def timestamp(value):
    if type(value) is not str or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise Blocked("INVALID_METADATA")
    try:
        dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise Blocked("INVALID_METADATA") from None
    return value


def boolean(value):
    if type(value) is not bool:
        raise Blocked("INVALID_METADATA")
    return value


def labels_have_plus(labels):
    if type(labels) is not list or len(labels) > 100:
        raise Blocked("INVALID_METADATA")
    for label in labels:
        if type(label) is not dict or type(label.get("name")) is not str:
            raise Blocked("INVALID_METADATA")
    return any(label["name"] == "+" for label in labels)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Blocked("HTTP_ERROR")


class APIReader:
    def __init__(self, token=None):
        self.token = token
        self.requests = 0
        self.open = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        ).open

    def __call__(self, path, **params):
        # Inputs below are constructed by this file, never taken from API URLs.
        allowed = (
            rf"{ROOT}/pulls(?:/[1-9][0-9]*)?",
            rf"{ROOT}/commits/[0-9a-f]{{40}}/(?:check-runs|status)",
            rf"{ROOT}/actions/(?:workflows|runs)",
            rf"{ROOT}/actions/runs/[1-9][0-9]*/jobs",
            rf"{ROOT}/rules/branches/main",
        )
        if not any(re.fullmatch(p, path) for p in allowed):
            raise Blocked("INVALID_METADATA")
        if self.requests >= 500:
            raise Blocked("LIMIT_REACHED")
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "unsloth-metadata-intake/1", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
        self.requests += 1
        try:
            with self.open(urllib.request.Request(url, headers=headers, method="GET"), timeout=25) as response:
                if response.status != 200:
                    raise Blocked("HTTP_ERROR")
                data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise Blocked("RESPONSE_TOO_LARGE")
            return json.loads(data)
        except urllib.error.HTTPError as exc:
            code = "HTTP_" + str(exc.code)
            raise Blocked(code if code in ERROR_CODES else "HTTP_ERROR") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise Blocked("NETWORK_UNAVAILABLE") from None
        except (ValueError, UnicodeError, RecursionError):
            raise Blocked("INVALID_METADATA") from None


def pages(api, path, key=None, total=None, **params):
    rows = []
    declared = None
    for page in range(1, MAX_PAGES + 1):
        response = api(path, per_page=100, page=page, **params)
        if key is None:
            batch = response
        else:
            if type(response) is not dict:
                raise Blocked("INVALID_METADATA")
            batch = response.get(key)
            if total:
                count = integer(response.get(total), maximum=100000)
                if declared is not None and count != declared:
                    raise Blocked("INCOMPLETE_PAGINATION")
                declared = count
        if type(batch) is not list or len(batch) > 100:
            raise Blocked("INVALID_METADATA")
        rows.extend(batch)
        if len(batch) < 100:
            if declared is not None and len(rows) != declared:
                raise Blocked("INCOMPLETE_PAGINATION")
            return rows
    raise Blocked("INCOMPLETE_PAGINATION")


def candidate(row):
    if type(row) is not dict or row.get("state") not in {"open", "closed"}:
        raise Blocked("INVALID_METADATA")
    base = row.get("base", {})
    head = row.get("head", {})
    if type(base) is not dict or type(head) is not dict or type(base.get("repo")) is not dict:
        raise Blocked("INVALID_METADATA")
    if base["repo"].get("id") != REPO_ID:
        raise Blocked("INVALID_METADATA")
    merge_sha = row.get("merge_commit_sha")
    return {
        "number": integer(row.get("number"), 1),
        "open": row["state"] == "open",
        "draft": boolean(row.get("draft")),
        "has_plus": labels_have_plus(row.get("labels")),
        "created_at": timestamp(row.get("created_at")),
        "head_sha": sha(head.get("sha")),
        "base_sha": sha(base.get("sha")),
        "base_is_main": base.get("ref") == "main",
        "test_merge_sha": sha(merge_sha) if merge_sha is not None else None,
    }


def select(api):
    found, seen, last_date = [], set(), "9999"
    for page in range(1, MAX_PAGES + 1):
        batch = api(ROOT + "/pulls", state="open", sort="created", direction="desc", per_page=100, page=page)
        if type(batch) is not list or len(batch) > 100:
            raise Blocked("INVALID_METADATA")
        for raw in batch:
            row = candidate(raw)
            if row["number"] in seen or row["created_at"] > last_date or not row["open"]:
                raise Blocked("INCOMPLETE_PAGINATION")
            seen.add(row["number"])
            last_date = row["created_at"]
            if not row["has_plus"]:
                found.append(row)
                if len(found) == LIMIT:
                    return found
        if len(batch) < 100:
            return found
    raise Blocked("INCOMPLETE_PAGINATION")


def check_summary(api, commit):
    checks = pages(api, ROOT + "/commits/" + commit + "/check-runs", "check_runs", "total_count", filter="latest")
    statuses = pages(api, ROOT + "/commits/" + commit + "/status", "statuses", "total_count")
    counts = {"success": 0, "failed": 0, "pending": 0, "not_success": 0}
    seen = set()
    for check in checks:
        if type(check) is not dict:
            raise Blocked("INVALID_METADATA")
        cid = integer(check.get("id"), 1)
        if cid in seen or sha(check.get("head_sha")) != commit:
            raise Blocked("INCOMPLETE_PAGINATION")
        seen.add(cid)
        status, conclusion = check.get("status"), check.get("conclusion")
        if status not in STATUSES or (conclusion is not None and conclusion not in CONCLUSIONS):
            raise Blocked("INVALID_METADATA")
        if status != "completed":
            counts["pending"] += 1
        elif conclusion == "success":
            counts["success"] += 1
        elif conclusion in {"failure", "timed_out", "startup_failure"}:
            counts["failed"] += 1
        else:
            counts["not_success"] += 1
    for row in statuses:
        if type(row) is not dict:
            raise Blocked("INVALID_METADATA")
        state = row.get("state")
        if state == "success":
            counts["success"] += 1
        elif state == "pending":
            counts["pending"] += 1
        elif state in {"failure", "error"}:
            counts["failed"] += 1
        else:
            raise Blocked("INVALID_METADATA")
    counts["total"] = len(checks) + len(statuses)
    return counts


def security_observation(api, row, workflow_id):
    if workflow_id is None:
        return {"state": "WORKFLOW_NOT_FOUND", "run_id": None}
    runs = pages(api, ROOT + "/actions/runs", "workflow_runs", "total_count", event="pull_request", head_sha=row["head_sha"])
    matches = []
    for run in runs:
        if type(run) is not dict:
            raise Blocked("INVALID_METADATA")
        if run.get("workflow_id") != workflow_id or run.get("head_sha") != row["head_sha"]:
            continue
        linked = run.get("pull_requests")
        if type(linked) is not list:
            raise Blocked("INVALID_METADATA")
        if not any(type(p) is dict and p.get("number") == row["number"] for p in linked):
            continue
        matches.append(run)
    if not matches:
        return {"state": "NO_CURRENT_HEAD_RUN", "run_id": None}
    run = max(matches, key=lambda r: (integer(r.get("id"), 1), integer(r.get("run_attempt", 1), 1)))
    rid = integer(run["id"], 1)
    if run.get("status") not in STATUSES:
        raise Blocked("INVALID_METADATA")
    if run["status"] != "completed":
        return {"state": "PENDING", "run_id": rid}
    if run.get("conclusion") != "success":
        return {"state": "NOT_SUCCESS", "run_id": rid}
    # A successful run is explicitly NOT proof of underlying strict scanner success.
    return {"state": "GREEN_BADGE_NOT_STRICT_AUDIT_EVIDENCE", "run_id": rid}


def assess(api, initial, workflow_id):
    row = dict(initial)
    row.update({"ci_gate": "NOT_CHECKED", "security_observation": {"state": "NOT_CHECKED", "run_id": None}, "review": "NOT_STARTED", "source_fetched": False})
    try:
        live = candidate(api(ROOT + "/pulls/" + str(row["number"])))
        if not live["open"] or live["has_plus"] or live["draft"]:
            row["ci_gate"] = "INELIGIBLE"
            row["open"], row["draft"], row["has_plus"] = live["open"], live["draft"], live["has_plus"]
            return row
        if any(live[k] != row[k] for k in ("head_sha", "base_sha")):
            row["ci_gate"] = "REVISION_CHANGED"
            return row
        row["test_merge_sha"] = live["test_merge_sha"]
        if not row["base_is_main"]:
            row["ci_gate"] = "TARGET_POLICY_UNCONFIGURED"
            return row
        row["head_checks"] = check_summary(api, row["head_sha"])
        row["merge_checks"] = check_summary(api, row["test_merge_sha"]) if row["test_merge_sha"] and row["test_merge_sha"] != row["head_sha"] else None
        summaries = [row["head_checks"]] + ([row["merge_checks"]] if row["merge_checks"] else [])
        totals = {key: sum(s[key] for s in summaries) for key in ("success", "failed", "pending", "not_success", "total")}
        row["observed_checks"] = totals
        if totals["failed"]:
            row["ci_gate"] = "BLOCKED_FAILED"
        elif totals["pending"]:
            row["ci_gate"] = "BLOCKED_PENDING"
        elif totals["not_success"]:
            row["ci_gate"] = "BLOCKED_NON_SUCCESS"
        elif totals["total"] == 0:
            row["ci_gate"] = "BLOCKED_NO_CHECKS"
        else:
            row["ci_gate"] = "OBSERVED_GREEN_REQUIREMENTS_UNVERIFIED"
        try:
            row["security_observation"] = security_observation(api, row, workflow_id)
        except Blocked as exc:
            row["security_observation"] = {"state": "UNAVAILABLE", "run_id": None, "error": exc.code}
        final = candidate(api(ROOT + "/pulls/" + str(row["number"])))
        if any(final[k] != row[k] for k in ("head_sha", "base_sha", "test_merge_sha", "open", "draft", "has_plus")):
            row["ci_gate"] = "REVISION_OR_ELIGIBILITY_CHANGED"
    except Blocked as exc:
        row["ci_gate"] = "BLOCKED_METADATA_ERROR"
        row["error"] = exc.code
    return row


def collect(api):
    result = {"schema": "unsloth-metadata-intake-v1", "repository": REPO, "repository_id": REPO_ID,
              "selection": "latest_20_open_without_exact_plus_by_creation", "collected_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "source_files_fetched": 0, "diffs_fetched": 0, "comments_fetched": 0, "security_scans_run": 0, "reviews_completed": 0, "candidate_writes": 0,
              "status": "BLOCKED", "candidates": []}
    try:
        selected = select(api)
        result["selected_count"] = len(selected)
        workflow_id = None
        try:
            workflows = pages(api, ROOT + "/actions/workflows", "workflows", "total_count")
            ids = [integer(w.get("id"), 1) for w in workflows if type(w) is dict and w.get("path") == ".github/workflows/security-audit.yml" and w.get("state") == "active"]
            if len(ids) == 1:
                workflow_id = ids[0]
        except Blocked as exc:
            result["catalog_error"] = exc.code
        for item in selected:
            result["candidates"].append(assess(api, item, workflow_id))
        try:
            latest = select(api)
            result["selection_still_latest"] = [r["number"] for r in latest] == [r["number"] for r in selected]
        except Blocked:
            result["selection_still_latest"] = False
        result["status"] = "METADATA_ASSESSED_REVIEW_GATES_CLOSED"
    except Blocked as exc:
        result["error"] = exc.code
    result["completed_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


def main():
    # Read only the explicitly provisioned runner token, never search for credentials.
    reader = APIReader(os.environ.get("GH_TOKEN"))
    try:
        output = collect(reader)
    except Exception:
        output = {"schema": "unsloth-metadata-intake-v1", "status": "BLOCKED", "error": "INVALID_METADATA", "candidates": []}
    output["api_requests"] = reader.requests
    commit, run, attempt = os.environ.get("GITHUB_SHA"), os.environ.get("GITHUB_RUN_ID"), os.environ.get("GITHUB_RUN_ATTEMPT")
    if commit and run and attempt:
        try:
            output["collector"] = {"commit": sha(commit), "run_id": integer(int(run), 1), "attempt": integer(int(attempt), 1)}
        except (ValueError, Blocked):
            output = {"schema": "unsloth-metadata-intake-v1", "status": "BLOCKED", "error": "INVALID_METADATA", "candidates": []}
    directory = Path.cwd() / "temp" / "intake-output"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "result.json"
    with target.open("x", encoding="utf-8") as f:
        json.dump(output, f, sort_keys=True, indent=2)
        f.write("\n")
    print("INTAKE_RESULT_WRITTEN")

if __name__ == "__main__":
    main()
