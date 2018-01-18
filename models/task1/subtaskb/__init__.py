"""
    This module provides the task1, subtask B models.
"""

from .base import Model
from .dummy import BEST, WORST, RANDOM, CONSERVATIVE
from .classifiers import REGRESSOR_CLASSIFIERS

DUMMY = [BEST, WORST, RANDOM, CONSERVATIVE]
CLASSIFIERS = REGRESSOR_CLASSIFIERS

MODELS = DUMMY + CLASSIFIERS
