# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Signed links carry their own account, and die with it.

These are the routes with no auth dependency: an HMAC token stands in for the bearer so a
browser can range-request a PDF or drop a URL into an <img>. The auth dependency is also
the only thing that binds an account, so each of these has to resolve the account from the
signed target itself, and re-check that the account is still there.
"""

import secrets

import pytest
from fastapi import HTTPException
from PIL import Image

from auth import policy, storage
from core.inference import image_gallery, video_gallery
from hub.services.models import account_access as access
from utils.account_context import OWNER, run_as

ALICE_NAME = "alice"


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    # Bumps the account generation, which is what every per-account link cache is keyed on.
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    alice = storage.issue_account_setup_code(username = ALICE_NAME)["account"]
    yield storage.get_account(ALICE_NAME), alice["account_id"]
    policy.invalidate_account_cache()


def test_a_signed_media_link_dies_with_its_account(accounts):
    """The token is still validly signed; nothing consulted the account behind it."""
    alice, account_id = accounts
    meta = {"prompt": "p", "model": "m", "created_at": 100.0, "width": 8, "height": 8}
    image_id = run_as(alice, image_gallery.save, Image.new("RGB", (8, 8)), meta)["id"]
    video_id = run_as(
        alice,
        video_gallery.save,
        b"\x00\x00\x00\x18ftypmp42",
        {**meta, "num_frames": 1, "fps": 1, "duration_s": 1.0},
    )["id"]

    for media_id in (image_id, video_id):
        target = run_as(alice, access.media_link_target, media_id)
        assert access.media_link_account(target, media_id).account_id == account_id

    storage.set_account_active(account_id, False)
    for media_id in (image_id, video_id):
        target = f"{account_id}:{media_id}"
        assert access.media_link_account(target, media_id) is None
    # The owner's own links are unchanged.
    assert access.media_link_account(image_id, image_id) is OWNER
