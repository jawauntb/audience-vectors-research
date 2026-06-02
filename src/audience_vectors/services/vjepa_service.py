"""Async wrapper around the V-JEPA Modal predictor.

Same two-tier error model as TribeService: validation errors propagate
(caller-fixable), everything else soft-fails to None.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VjepaValidationError(ValueError):
    """Bad input to V-JEPA — fix the caller, not the predictor."""


_VALIDATION_PREFIXES = (
    "video too long",
    "unsupported scheme",
    "remote media too large",
)


def _is_validation_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _VALIDATION_PREFIXES)


class VjepaService:
    """Async wrapper that dispatches video stimuli to the V-JEPA Modal predictor.

    Looks up the deployed `VjepaPredictor` class by app name so this works
    from a local script. Override `app_name` if you've deployed under
    a different environment suffix.
    """

    def __init__(self, app_name: str | None = None) -> None:
        from audience_vectors.modal_app.app import get_app_name

        self.app_name = app_name or get_app_name()
        self._cls: Any = None

    def _resolve_cls(self) -> Any:
        if self._cls is not None:
            return self._cls
        import modal  # noqa: PLC0415

        self._cls = modal.Cls.from_name(self.app_name, "VjepaPredictor")
        return self._cls

    async def predict_video(self, video_path_or_url: str) -> Any:
        if not video_path_or_url:
            return None
        try:
            cls = self._resolve_cls()
            predictor = cls()
            return await predictor.predict_video.remote.aio(video_path_or_url)
        except Exception as exc:  # noqa: BLE001
            if _is_validation_error(exc):
                raise VjepaValidationError(str(exc)) from exc
            logger.exception("V-JEPA predict_video failed: %s", exc)
            return None

    async def predict_video_bytes(
        self, video_bytes: bytes, suffix: str = ".mp4"
    ) -> Any:
        if not video_bytes:
            return None
        try:
            cls = self._resolve_cls()
            predictor = cls()
            return await predictor.predict_video_bytes.remote.aio(video_bytes, suffix)
        except Exception as exc:  # noqa: BLE001
            if _is_validation_error(exc):
                raise VjepaValidationError(str(exc)) from exc
            logger.exception("V-JEPA predict_video_bytes failed: %s", exc)
            return None
