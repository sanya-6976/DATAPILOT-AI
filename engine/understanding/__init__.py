"""Validated contracts for dataset understanding and evidence provenance."""

from engine.understanding.evaluation import (
    EvaluationMetrics,
    score_labeled_outputs,
)
from engine.understanding.evidence import (
    ColumnIdentity,
    EvidenceAdapterResult,
    adapt_profile_and_quality_evidence,
)
from engine.understanding.field_roles import infer_field_roles
from engine.understanding.models import (
    AnalysisScope,
    ColumnUnderstanding,
    ConfidenceEstimate,
    DatasetUnderstandingResult,
    EvidenceItem,
    Hypothesis,
    PairwiseEvidence,
    RelationshipHypothesis,
    UnderstandingWarning,
)
from engine.understanding.pairwise import (
    PairwiseEvidenceLimits,
    PairwiseEvidenceResult,
    build_pairwise_evidence,
)
from engine.understanding.pipeline import run_understanding_pipeline
from engine.understanding.relationships import (
    discover_relationships,
    discover_relationships_from_dataframe,
)
from engine.understanding.semantic_types import infer_semantic_types
from engine.understanding.value_interpretations import (
    infer_value_interpretations,
)

__all__ = [
    "AnalysisScope",
    "ColumnIdentity",
    "ColumnUnderstanding",
    "ConfidenceEstimate",
    "DatasetUnderstandingResult",
    "EvaluationMetrics",
    "EvidenceAdapterResult",
    "EvidenceItem",
    "Hypothesis",
    "PairwiseEvidence",
    "PairwiseEvidenceLimits",
    "PairwiseEvidenceResult",
    "RelationshipHypothesis",
    "UnderstandingWarning",
    "adapt_profile_and_quality_evidence",
    "build_pairwise_evidence",
    "discover_relationships",
    "discover_relationships_from_dataframe",
    "infer_field_roles",
    "infer_semantic_types",
    "infer_value_interpretations",
    "run_understanding_pipeline",
    "score_labeled_outputs",
]
