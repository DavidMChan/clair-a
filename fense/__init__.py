"""
FENSE: Fluency and Semantic Evaluation package.

Vendored and updated version for clair-a compatibility.
"""

from .evaluator import Evaluator
from .model import BERTFlatClassifier

__version__ = "0.2.0-vendored"
__all__ = ["BERTFlatClassifier", "Evaluator"]
