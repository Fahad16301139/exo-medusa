"""
Medusa inference engine for EXO framework.
This module provides an inference engine that uses the Medusa model for parallel decoding.
"""

import torch
from typing import Optional, Dict, Any

from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from .medusa_model import MedusaShardedModel

class MedusaInferenceEngine(InferenceEngine):
    """
    Inference engine that uses Medusa for parallel decoding.
    This class inherits from the base InferenceEngine and adds Medusa-specific functionality.
    """
    
    def __init__(self, shard_downloader: ShardDownloader, medusa_num_heads: int = 5, medusa_num_layers: int = 1):
        super().__init__(shard_downloader)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.medusa_initialized = False
        
    async def ensure_shard(self, shard: Shard):
        """
        Override ensure_shard to initialize Medusa components.
        """
        await super().ensure_shard(shard)
        
        if not self.medusa_initialized and isinstance(self.sharded_model, MedusaShardedModel):
            self.sharded_model.initialize_medusa(
                medusa_num_heads=self.medusa_num_heads,
                medusa_num_layers=self.medusa_num_layers
            )
            self.medusa_initialized = True
            
    def create_sharded_model(self, *args, **kwargs) -> MedusaShardedModel:
        """
        Override to create a MedusaShardedModel instead of the standard ShardedGeneralModel.
        """
        return MedusaShardedModel(*args, **kwargs)
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Override generate to use Medusa's parallel decoding when available.
        """
        if not self.medusa_initialized:
            # Fall back to standard generation if Medusa is not initialized
            return await super().generate(prompt, **kwargs)
            
        # Use Medusa's generation with the configured parameters
        return await self._generate_with_medusa(prompt, **kwargs)
        
    async def _generate_with_medusa(self, prompt: str, **kwargs) -> str:
        """
        Internal method to handle generation with Medusa's parallel decoding.
        """
        # Ensure we have a shard
        if self.shard is None:
            raise RuntimeError("No shard selected")
            
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate with Medusa
        output_ids = self.sharded_model.generate(
            input_ids,
            max_length=kwargs.get("max_length", 100),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            **kwargs
        )
        
        # Decode and return
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True) 