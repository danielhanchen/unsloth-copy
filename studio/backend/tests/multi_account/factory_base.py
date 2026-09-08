# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared factory vocabulary for the route isolation matrix.

A factory names a seeder that creates the resource inside the acting account and returns the
path parameters the route needs. Actor expectations default to the isolation contract: the
owner and the other account get 404, the resource's account gets the success code, an
unauthenticated caller gets 401/403 and a deactivated account gets 401. Any route whose real
contract differs states the deviation together with a one-line reason.
"""

import re
from dataclasses import dataclass, field
from typing import Callable

# Seeder name -> callable(account) -> path parameters for the routes that use it.
SEEDERS: dict[str, Callable] = {}

_CONVERTER = re.compile(r"\{([^}:]+):[^}]+\}")


def seeder(name: str):
    """Register a resource seeder under ``name`` for use by Factory(name = ...)."""

    def decorate(function: Callable) -> Callable:
        assert name not in SEEDERS, f"duplicate seeder: {name}"
        SEEDERS[name] = function
        return function

    return decorate


def format_path(path: str, params: dict) -> str:
    """Fill a router path, dropping Starlette converters such as ``{filename:path}``."""
    return _CONVERTER.sub(r"{\1}", path).format(**params)


@dataclass(frozen = True)
class Factory:
    name: str
    body: dict | list | None = None
    success: int = 200
    fragment: str | None = None
    owner: tuple[int, ...] = (404,)
    wrong: tuple[int, ...] = (404,)
    unauthenticated: tuple[int, ...] = (401, 403)
    deactivated: tuple[int, ...] = (401,)
    right: tuple[int, ...] | None = None
    self_expected: tuple[int, ...] | None = None
    reason: str = ""
    query: dict = field(default_factory = dict)
    extra_params: dict = field(default_factory = dict)

    def expected(self, actor: str) -> tuple[int, ...]:
        return {
            "owner": self.owner,
            "right": self.right or (self.success,),
            "wrong": self.wrong,
            "unauthenticated": self.unauthenticated,
            "deactivated": self.deactivated,
        }[actor]

    @property
    def deviates(self) -> bool:
        """True when this route does not follow the plain owner/other-account 404 contract."""
        return (
            self.owner != (404,)
            or self.wrong != (404,)
            or self.right is not None
            or self.self_expected is not None
        )


def merge(*tables: dict) -> dict:
    """Union of per-domain factory tables, rejecting a key claimed twice."""
    merged: dict = {}
    for table in tables:
        for key, value in table.items():
            assert key not in merged, f"duplicate factory key: {key}"
            merged[key] = value
    return merged
