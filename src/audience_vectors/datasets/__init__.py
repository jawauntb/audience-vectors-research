"""Dataset adapters. Each adapter yields `CanonicalVideo` rows from one source."""

from audience_vectors.datasets.base import DatasetAdapter
from audience_vectors.datasets.bold_moments import BoldMomentsAdapter
from audience_vectors.datasets.memento10k import Memento10kAdapter

__all__ = ["BoldMomentsAdapter", "DatasetAdapter", "Memento10kAdapter"]
