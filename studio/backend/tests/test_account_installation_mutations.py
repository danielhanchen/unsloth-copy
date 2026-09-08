# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installation-wide mutations stay with the installation owner."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import get_current_subject
from utils.account_context import AccountContext, OWNER, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def multi_user(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    yield


def _client(account, build):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    build(app)
    return TestClient(app)


def _llama_app(app):
    from routes import llama

    app.include_router(llama.router, prefix = "/api/llama")


def _shutdown_route():
    import main

    for route in main.app.routes:
        if getattr(route, "path", None) == "/api/shutdown" and "POST" in getattr(
            route, "methods", ()
        ):
            return route
    raise AssertionError("POST /api/shutdown is not registered")


@pytest.mark.parametrize("path,payload", [("/api/llama/update", None), ("/api/llama/backend", {"backend": "cpu"})])
def test_a_managed_account_cannot_replace_the_installation_executables(
    monkeypatch, path, payload
):
    from utils import llama_cpp_update

    started = []
    monkeypatch.setattr(
        llama_cpp_update, "start_update", lambda: started.append("update") or {"started": True}
    )
    monkeypatch.setattr(
        llama_cpp_update,
        "start_backend_switch",
        lambda backend: started.append(backend) or {"started": True},
    )
    from routes import llama as llama_routes

    monkeypatch.setattr(llama_routes, "start_update", llama_cpp_update.start_update)
    monkeypatch.setattr(llama_routes, "start_backend_switch", llama_cpp_update.start_backend_switch)

    with _client(ALICE, _llama_app) as client:
        assert client.post(path, json = payload).status_code == 403
    assert started == []
    with _client(OWNER, _llama_app) as client:
        assert client.post(path, json = payload).status_code == 200
    assert started


def test_a_managed_account_cannot_stop_the_shared_server():
    """Shutdown is process-wide, so it is solved through require_owner before it is served."""
    route = _shutdown_route()
    solved = [
        dependency.call
        for dependency in route.dependant.dependencies
        if dependency.call is not None
    ]
    assert policy.require_owner in solved
    # And the subject is bound first, or require_owner would judge an unbound context.
    assert solved.index(get_current_subject) < solved.index(policy.require_owner)
