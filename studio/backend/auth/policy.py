# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installation-wide account policy: one-user install (nothing changes) versus
multi-user (isolation applies).

Answers are cached against an account generation bumped on create/delete, so a
one-user install queries nothing.
"""

from __future__ import annotations

import threading
from typing import Optional

from fastapi import Depends, HTTPException, status

LOGIN_MODE_SINGLE = "single"
LOGIN_MODE_MULTI = "multi"

_lock = threading.Lock()
_generation = 0
_cached: Optional[tuple[int, int, int]] = None  # (generation, active accounts, managed accounts)


def invalidate_account_cache() -> None:
    """Call after any account is created, deleted, activated or deactivated."""
    global _generation, _cached
    with _lock:
        _generation += 1
        _cached = None


def _account_counts() -> tuple[int, int]:
    global _cached
    with _lock:
        if _cached is not None and _cached[0] == _generation:
            return _cached[1], _cached[2]
        generation = _generation
    from auth import storage

    try:
        active, managed = storage.account_counts()
    except Exception:  # noqa: BLE001 - an unreadable auth.db is a one-user install
        # Single login form, but count a managed account so the host stays closed.
        active, managed = 1, 1
    with _lock:
        if generation == _generation:
            _cached = (generation, active, managed)
    return active, managed


def active_account_count() -> int:
    return _account_counts()[0]


def managed_account_count() -> int:
    """Managed accounts of any state; a deactivated one still counts, its files remain on disk."""
    return _account_counts()[1]


def installation_is_multi_user() -> bool:
    return active_account_count() > 1


def login_mode() -> str:
    return LOGIN_MODE_MULTI if installation_is_multi_user() else LOGIN_MODE_SINGLE


def installation_has_managed_accounts() -> bool:
    """Whether any managed account exists, active or not.

    Gates that hand out the owner's identity without a credential close on this, not on
    the login mode: a deactivated account's files stay on disk until it is deleted.
    """
    return installation_is_multi_user() or managed_account_count() > 0


def full_access_permitted() -> bool:
    """Whether the unsandboxed tool modes may run at all.

    Full access reads and writes the host as the server user, so it is refused install-wide
    (owner included) as soon as any managed account exists, even a deactivated one.
    """
    return not installation_has_managed_accounts()


def _forbid(detail: str) -> HTTPException:
    return HTTPException(status_code = status.HTTP_403_FORBIDDEN, detail = detail)


async def require_owner() -> None:
    """Dependency for installation-wide operations: accounts, updates, shutdown,
    executables, network and caches."""
    from utils.account_context import current_account
    if not current_account().is_owner:
        raise _forbid("Only the installation owner can do this")


def require_account_scope(resource_account_id: Optional[str]) -> None:
    """Refuse a resource owned by another account; ``None`` predates accounts and is the owner's."""
    from utils.account_context import OWNER_ACCOUNT_ID, current_account_id

    owner_of = resource_account_id or OWNER_ACCOUNT_ID
    if owner_of != current_account_id():
        # 404 rather than 403: the resource's existence is itself information.
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = "Not found")


OwnerOnly = Depends(require_owner)
