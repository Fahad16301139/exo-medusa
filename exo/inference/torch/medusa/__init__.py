"""
Medusa integration for EXO framework.
This module provides Medusa model support for the EXO inference engine.
"""

__version__ = "0.1.0"

# Import here to avoid circular imports
from .medusa_model import MedusaShardedModel
from .medusa_inference_engine import MedusaInferenceEngine

__all__ = ["MedusaShardedModel", "MedusaInferenceEngine"] 