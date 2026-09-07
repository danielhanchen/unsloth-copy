# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's recipe MCP providers obey the chat MCP network boundary.

Chat validates every resolved MCP address and pins the connection to it, so a
managed account cannot reach loopback or the LAN. The Data Designer engine opens
its own connections, so a recipe provider used to be built from an endpoint chat
refuses outright.
"""

from __future__ import annotations

import sys
import types

import pytest
from fastapi import HTTPException

from auth import policy
from core.inference import mcp_client
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("alice-id", "alice")

_PRIVATE_ENDPOINTS = (
    "http://127.0.0.1:9111/mcp",
    "http://localhost:9111/sse",
    "http://10.0.0.5:8000/mcp",
)


@pytest.fixture(autouse = True)
def data_designer_stub(monkeypatch, tmp_path):
    """The recipe engine is an Unsloth-only plugin that is not installed on CI, and
    ``build_mcp_providers`` imports its provider classes lazily. Stand in for them so
    the account boundary is exercised without it."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)

    class _Provider:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    package = types.ModuleType("data_designer")
    config = types.ModuleType("data_designer.config")
    mcp = types.ModuleType("data_designer.config.mcp")
    mcp.MCPProvider = type("MCPProvider", (_Provider,), {})
    mcp.LocalStdioMCPProvider = type("LocalStdioMCPProvider", (_Provider,), {})
    config.mcp = mcp
    package.config = config
    for name, module in (
        ("data_designer", package),
        ("data_designer.config", config),
        ("data_designer.config.mcp", mcp),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _recipe(endpoint: str) -> dict:
    return {
        "mcp_providers": [
            {"provider_type": "streamable_http", "name": "local", "endpoint": endpoint}
        ]
    }


@pytest.mark.parametrize("endpoint", _PRIVATE_ENDPOINTS)
def test_managed_recipe_mcp_is_refused_where_chat_mcp_refuses(endpoint):
    from core.data_recipe.service import build_mcp_providers

    with pytest.raises(HTTPException) as chat:
        run_as(ALICE, mcp_client.validate_mcp_address, endpoint)
    assert chat.value.status_code == 400
    with pytest.raises(HTTPException) as recipe:
        run_as(ALICE, build_mcp_providers, _recipe(endpoint))
    assert recipe.value.status_code == 403


def test_owner_recipe_mcp_is_unchanged():
    from core.data_recipe.service import build_mcp_providers

    built = run_as(OWNER, build_mcp_providers, _recipe(_PRIVATE_ENDPOINTS[0]))
    assert [provider.endpoint for provider in built] == [_PRIVATE_ENDPOINTS[0]]


def test_single_account_installs_keep_recipe_mcp(monkeypatch):
    from core.data_recipe.service import build_mcp_providers

    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    built = run_as(ALICE, build_mcp_providers, _recipe(_PRIVATE_ENDPOINTS[0]))
    assert [provider.endpoint for provider in built] == [_PRIVATE_ENDPOINTS[0]]


def test_managed_recipe_stdio_provider_is_still_dropped(monkeypatch):
    """The host gate already refuses a managed account a local command; the network
    refusal must not turn that silent drop into an error."""
    from core.data_recipe.service import build_mcp_providers

    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    recipe = {
        "mcp_providers": [
            {"provider_type": "stdio", "name": "fs", "command": "npx", "args": [], "env": {}}
        ]
    }
    assert run_as(ALICE, build_mcp_providers, recipe) == []
