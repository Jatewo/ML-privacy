"""Attack modules.

This module provides:
- MembershipInferenceAttack: Abstract base class for attacks.
- LossBasedMIA: Attack based on the loss values.
"""
from .membership_inference_attack import MembershipInferenceAttack
from .lossbased_attack import LossBasedMIA

__all__ = ["MembershipInferenceAttack", "LossBasedMIA"]
