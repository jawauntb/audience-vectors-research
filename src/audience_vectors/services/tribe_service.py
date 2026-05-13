"""Async service wrapper around the TRIBE Modal predictor.

Mirrors the superoptimizers `NeuralEngagementService` pattern but strips
production observability and tiered error handling — research code, not
prod traffic. Two-tier error model:

  - `TribeValidationError`: caller-fixable (bad URL, too long, etc.). Re-raise.
  - All others: log and return None so a single bad clip doesn't kill a batch.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TribeValidationError(ValueError):
    """Bad input to TRIBE — fix the caller, not the predictor."""


class TribeBackendError(RuntimeError):
    """Modal/TRIBE blew up in a way the caller can't fix."""


# Prefixes the Modal worker uses in raised exception messages; recognize
# them on the service side because Modal serializes exceptions across
# its RPC boundary and the custom classes aren't importable here.
_VALIDATION_PREFIXES = (
    "video too long",
    "unsupported URI scheme",
    "remote media too large",
    "remote media exceeded size cap",
    "non-positive video duration",
)


def _is_validation_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _VALIDATION_PREFIXES)


class TribeService:
    """Async wrapper that dispatches video stimuli to the Modal predictor.

    Looks up the deployed `TribeV2Predictor` class by app name so the
    service works from a local script (not just inside Modal containers).
    """

    def __init__(self, app_name: str | None = None) -> None:
        from audience_vectors.modal_app.app import get_app_name

        self.app_name = app_name or get_app_name()
        self._cls: Any = None

    def _resolve_cls(self) -> Any:
        if self._cls is not None:
            return self._cls
        import modal  # noqa: PLC0415

        self._cls = modal.Cls.from_name(self.app_name, "TribeV2Predictor")
        return self._cls

    async def predict_video(self, video_path_or_url: str) -> Any:
        """Run TRIBE v2 on one video. Returns VideoPredictionResult or None on soft failure."""
        if not video_path_or_url:
            return None
        try:
            cls = self._resolve_cls()
            predictor = cls()
            return await predictor.predict_video.remote.aio(video_path_or_url)
        except Exception as exc:  # noqa: BLE001
            if _is_validation_error(exc):
                raise TribeValidationError(str(exc)) from exc
            logger.exception("TRIBE predict_video failed: %s", exc)
            return None
