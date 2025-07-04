"""
Cardiac Digital Twins Enhanced - A comprehensive framework for physics-informed 
medical digital twins using self-supervised learning.

This package provides enhanced implementations of the Med-Real2Sim methodology
with improved code organization, documentation, and additional features.
"""

__version__ = "1.0.0"
__author__ = "Enhanced Med-Real2Sim Team"
__email__ = "contact@cardiac-digital-twins.org"

from . import physics
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = [
    "physics",
    "models", 
    "training",
    "evaluation",
    "utils"
]

