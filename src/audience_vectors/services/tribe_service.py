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
    "text produced no retained tribe segments",
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

    async def predict_text(self, text_path_or_url: str) -> Any:
        """Run TRIBE v2 on one text stimulus. Returns VideoPredictionResult or None."""
        if not text_path_or_url:
            return None
        try:
            cls = self._resolve_cls()
            predictor = cls()
            return await predictor.predict_text.remote.aio(text_path_or_url)
        except Exception as exc:  # noqa: BLE001
            if _is_validation_error(exc):
                raise TribeValidationError(str(exc)) from exc
            logger.exception("TRIBE predict_text failed: %s", exc)
            return None

    async def predict_video_time_pos_scale(
        self,
        video_path_or_url: str,
        time_pos_scale: float,
    ) -> Any:
        """Run TRIBE with its learned temporal position embedding scaled."""
        if not video_path_or_url:
            return None
        try:
            cls = self._resolve_cls()
            predictor = cls()
            return await predictor.predict_video_time_pos_scale.remote.aio(
                video_path_or_url,
                time_pos_scale,
            )
        except Exception as exc:  # noqa: BLE001
            if _is_validation_error(exc):
                raise TribeValidationError(str(exc)) from exc
            logger.exception("TRIBE time_pos_scale predict failed: %s", exc)
            return None

    async def predict_video_hidden_patch(
        self,
        video_path_or_url: str,
        *,
        hook_module: str = "_model.encoder",
        patch_mode: str = "none",
        patch_scale: float = 1.0,
        rotary_inv_freq_scale: float = 1.0,
        capture_hidden: bool = False,
    ) -> Any:
        """Run TRIBE with an optional hidden-state or rotary-position patch."""
        if not video_path_or_url:
            return None
        try:
            cls = self._resolve_cls()
            predictor = cls()
            return await predictor.predict_video_hidden_patch.remote.aio(
                video_path_or_url,
                hook_module,
                patch_mode,
                patch_scale,
                rotary_inv_freq_scale,
                capture_hidden,
            )
        except Exception as exc:  # noqa: BLE001
            if _is_validation_error(exc):
                raise TribeValidationError(str(exc)) from exc
            logger.exception("TRIBE hidden patch predict failed: %s", exc)
            return None

    async def capture_video_hiddens(
        self,
        video_path_or_url: str,
        hook_modules: list[str],
    ) -> Any:
        """Run TRIBE once and capture hidden tensors from several hook modules."""
        if not video_path_or_url:
            return None
        try:
            cls = self._resolve_cls()
            predictor = cls()
            return await predictor.capture_video_hiddens.remote.aio(
                video_path_or_url,
                hook_modules,
            )
        except Exception as exc:  # noqa: BLE001
            if _is_validation_error(exc):
                raise TribeValidationError(str(exc)) from exc
            logger.exception("TRIBE hidden capture failed: %s", exc)
            return None

    async def predict_video_hidden_direction_patch(
        self,
        video_path_or_url: str,
        *,
        hook_module: str,
        direction_npz: bytes,
        patch_alpha: float = 1.0,
    ) -> Any:
        """Run TRIBE after patching one learned hidden direction."""
        if not video_path_or_url:
            return None
        try:
            cls = self._resolve_cls()
            predictor = cls()
            return await predictor.predict_video_hidden_direction_patch.remote.aio(
                video_path_or_url,
                hook_module,
                direction_npz,
                patch_alpha,
            )
        except Exception as exc:  # noqa: BLE001
            if _is_validation_error(exc):
                raise TribeValidationError(str(exc)) from exc
            logger.exception("TRIBE hidden direction patch failed: %s", exc)
            return None
