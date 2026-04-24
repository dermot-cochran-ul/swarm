"""
EPISTEME: An Epistemic Foundation Model

A foundation model architecture that treats beliefs, uncertainty, evidence,
and revision as first-class system primitives.
"""

from episteme.models import Belief, BeliefType, BeliefRevision, Evidence
from episteme.core import EpistemicCore, UpdateEligibilityError
from episteme.memory import EpistemicMemory
from episteme.interface import LanguageInterface, Claim
from episteme.experience import ExperienceLoop

__all__ = [
    "Belief",
    "BeliefType",
    "BeliefRevision",
    "Evidence",
    "EpistemicCore",
    "UpdateEligibilityError",
    "EpistemicMemory",
    "LanguageInterface",
    "Claim",
    "ExperienceLoop",
]
