"""Contrastive coverage: identification and certification for policy differences."""

from .bandit import IdentifiedInterval
from .tree import contrastive_equivalence_classes, off_support_signatures

__all__ = [
    "IdentifiedInterval",
    "contrastive_equivalence_classes",
    "off_support_signatures",
]
