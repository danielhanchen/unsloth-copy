# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The whole account-policy table, in one place.

Two predicates ride on the same counts and they deliberately disagree. ``login_mode`` and
``installation_is_multi_user`` follow the number of ACTIVE accounts, so a deactivated one
keeps the single login form. ``installation_has_managed_accounts`` and the full-access gate
follow whether any managed account exists at all, active or not, because its files are still
on disk. The unreadable-database fallback splits them on purpose: one login form, closed host.

Five independent attempts at collapsing these predicates broke exactly this table, so the
four states are asserted together rather than one per test.
"""

import secrets

import pytest

from auth import policy, storage


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield storage
    policy.invalidate_account_cache()


def _state() -> tuple[str, bool, bool, bool]:
    policy.invalidate_account_cache()
    return (
        policy.login_mode(),
        policy.installation_is_multi_user(),
        policy.installation_has_managed_accounts(),
        policy.full_access_permitted(),
    )


def test_owner_only(auth_db):
    assert _state() == ("single", False, False, True)


def test_owner_plus_one_active_managed_account(auth_db):
    storage.issue_account_setup_code(username = "alice")
    assert _state() == ("multi", True, True, False)


def test_owner_plus_one_deactivated_managed_account(auth_db):
    account = storage.issue_account_setup_code(username = "alice")["account"]
    storage.set_account_active(account["account_id"], False)
    assert _state() == ("single", False, True, False)


def test_an_unreadable_auth_database(auth_db, monkeypatch):
    def boom():
        raise OSError("auth.db unreadable")

    monkeypatch.setattr(storage, "account_counts", boom)
    assert _state() == ("single", False, True, False)
