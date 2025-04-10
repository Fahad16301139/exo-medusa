"""
Torch inference engine module.
"""

from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine

# Export the inference engine
TorchInferenceEngine = TorchDynamicShardInferenceEngine

__all__ = ['TorchInferenceEngine']
