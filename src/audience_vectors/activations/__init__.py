"""Activation analysis — contrastive vectors, projections, patching, steering."""

from audience_vectors.activations.contrastive_vectors import (
    ContrastiveVectorTrainer,
    project_features,
)
from audience_vectors.activations.cross_validation import (
    CrossValSummary,
    FoldResult,
    cross_validate_contrastive,
    random_baseline_spearman,
    summary_to_dict,
)

__all__ = [
    "ContrastiveVectorTrainer",
    "CrossValSummary",
    "FoldResult",
    "cross_validate_contrastive",
    "project_features",
    "random_baseline_spearman",
    "summary_to_dict",
]
