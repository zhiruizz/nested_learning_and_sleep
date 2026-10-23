"""Partial reference primitives. No complete HOPE/Sleep model is implemented."""
from .memory import LinearSelfModifyingReference, dgd_linear_step
from .clocks import GradientWindow
from .consolidation import LinearExpertChain, ConsolidationTransaction

__all__ = ['LinearSelfModifyingReference', 'dgd_linear_step', 'GradientWindow',
           'LinearExpertChain', 'ConsolidationTransaction']
