"""Adapter interface. Implement one subclass per dataset (VideoMem, Memento10k, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from audience_vectors.schemas import CanonicalVideo


class DatasetAdapter(ABC):
    """Yield CanonicalVideo rows from a single source dataset.

    Subclasses are responsible for:
      1. Locating media files on disk (root dir comes from env / config).
      2. Reading the source dataset's labels + metadata.
      3. Mapping to the canonical schema (normalize labels later, not here).
    """

    #: Stable short name used as a column value, e.g. "VideoMem".
    name: str = ""

    #: Domain tag applied to every video from this adapter, e.g. "everyday".
    default_domain: str = "unknown"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.__class__.__name__}: dataset root does not exist: {self.root}"
            )

    @abstractmethod
    def iter_videos(self) -> Iterator[CanonicalVideo]:
        """Yield one CanonicalVideo per source video. No segmentation here."""

    def __iter__(self) -> Iterator[CanonicalVideo]:
        return self.iter_videos()
