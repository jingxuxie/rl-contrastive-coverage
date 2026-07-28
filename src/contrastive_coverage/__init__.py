"""Contrastive coverage: identification and certification for policy differences."""

from .bandit import IdentifiedInterval
from .efficient import (
    VarianceDecomposition,
    action_composition_noise,
    contrastive_augmented_contributions,
    empirical_pooled_action_means,
    empirical_stage_action_means,
    signed_pdis_contributions,
)
from .tabular import (
    SharedBranchTabularProcess,
    TabularModel,
    fit_tabular_model,
    make_stochastic_validation_process,
)
from .tree import contrastive_equivalence_classes, off_support_signatures

__all__ = [
    "IdentifiedInterval",
    "VarianceDecomposition",
    "action_composition_noise",
    "contrastive_augmented_contributions",
    "empirical_pooled_action_means",
    "empirical_stage_action_means",
    "signed_pdis_contributions",
    "SharedBranchTabularProcess",
    "TabularModel",
    "fit_tabular_model",
    "make_stochastic_validation_process",
    "contrastive_equivalence_classes",
    "off_support_signatures",
]
