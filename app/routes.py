from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter


def build_router(
    *,
    health_handler: Callable[[], dict[str, str]],
    reset_handler: Callable[[Any], Any],
    step_handler: Callable[[Any], Any],
    state_handler: Callable[[], Any],
) -> APIRouter:
    router = APIRouter()
    router.add_api_route("/health", health_handler, methods=["GET"])
    router.add_api_route("/reset", reset_handler, methods=["POST"])
    router.add_api_route("/step", step_handler, methods=["POST"])
    router.add_api_route("/state", state_handler, methods=["GET"])
    return router
