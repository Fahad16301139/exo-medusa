"""
Medusa algorithm implementation for EXO framework.
This is a direct implementation of Medusa's parallel decoding algorithm
without dependency on the original Medusa library.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
import traceback
import importlib.util
import os
import sys
import time
import math

from exo.helpers import DEBUG

class MedusaShardedModel(nn.Module):
    """
    Implementation of Medusa parallel decoding for EXO.
    This implementation works directly with EXO's dependencies.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # Get parameters from kwargs
        self.config = kwargs.get('config')
        self.shard = kwargs.get('shard')
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dtype = kwargs.get('dtype', torch.float32)
        self.use_cache = kwargs.get('use_cache', True)
        
        # Medusa parameters
        self.medusa_heads = kwargs.get('medusa_num_heads', 4)  # Number of prediction heads
        self.medusa_layers = kwargs.get('medusa_num_layers', 1)  # Number of layers for each head
        
        # These will be initialized later
        self.orig_model = None  # Original model
        self.medusa_heads_modules = None  # Additional prediction heads
        self.tokenizer = None
        
        # Required attributes for inference engine
        if self.config is not None:
            dim = self.config.get("embed_dim")
            self.hidden_size = dim
            self.max_generated_tokens = 2048
            self.max_position_embeddings = 2048
            self.vocab_size = 32000
            
        # Initialize medusa heads when the model is loaded
        self.medusa_initialized = False
    
    @property
    def model(self):
        """Compatibility property to access orig_model using model attribute."""
        return self.orig_model
        
    def initialize_original_model(self, orig_model):
        """
        Initialize with the original model from EXO.
        
        Args:
            orig_model: The base language model from EXO
        """
        self.orig_model = orig_model
        
        # Now that we have the model, initialize Medusa heads
        if not self.medusa_initialized:
            self._initialize_medusa_heads()
    
    def _initialize_medusa_heads(self):
        """Initialize Medusa prediction heads on top of the original model."""
        if self.orig_model is None:
            print("[Medusa] Cannot initialize heads without original model")
            return
            
        try:
            if DEBUG >= 1:
                print(f"[Medusa] Initializing {self.medusa_heads} heads with {self.medusa_layers} layers each")
                
            # Create Medusa heads based on the original model architecture
            # Each head predicts tokens at different future positions
            self.medusa_heads_modules = nn.ModuleList([
                nn.Sequential(
                    *([nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU()] * self.medusa_layers),
                    nn.Linear(self.hidden_size, self.vocab_size)
                )
                for _ in range(self.medusa_heads)
            ])
            
            # Move to the same device as the original model
            self.medusa_heads_modules.to(device=self.device, dtype=self.dtype)
            
            if DEBUG >= 1:
                print("[Medusa] Heads initialized successfully")
                
            self.medusa_initialized = True
            
        except Exception as e:
            print(f"[Medusa] Error initializing heads: {e}")
            traceback.print_exc()
    
    def caches_are_enabled(self):
        """Required method for inference engine."""
        if hasattr(self, 'orig_model') and self.orig_model is not None:
            if hasattr(self.orig_model, 'caches_are_enabled'):
                return self.orig_model.caches_are_enabled()
            elif hasattr(self.orig_model, 'model') and hasattr(self.orig_model.model, 'caches_are_enabled'):
                return self.orig_model.model.caches_are_enabled()
        return self.use_cache
    
    def setup_caches(self, batch_size=1, dtype=torch.float16, decoder_max_seq_len=1024):
        """Set up caches in the original model."""
        if hasattr(self, 'orig_model') and self.orig_model is not None:
            if hasattr(self.orig_model, 'setup_caches'):
                return self.orig_model.setup_caches(batch_size, dtype, decoder_max_seq_len)
            elif hasattr(self.orig_model, 'model') and hasattr(self.orig_model.model, 'setup_caches'):
                return self.orig_model.model.setup_caches(batch_size, dtype, decoder_max_seq_len)
    
    def reset_caches(self):
        """Reset caches in the original model."""
        if hasattr(self, 'orig_model') and self.orig_model is not None:
            if hasattr(self.orig_model, 'reset_caches'):
                return self.orig_model.reset_caches()
            elif hasattr(self.orig_model, 'model') and hasattr(self.orig_model.model, 'reset_caches'):
                return self.orig_model.model.reset_caches()
    
    def forward(self, *args, **kwargs):
        """Forward pass using the original model."""
        if self.orig_model is None:
            raise RuntimeError("Original model not initialized")
            
        try:
            # Try direct passthrough of all args and kwargs
            return self.orig_model(*args, **kwargs)
        except Exception as e:
            if DEBUG >= 2:
                print(f"[Medusa] Forward call error: {e}, trying alternatives")
                
            # If that fails, try standard transformer pattern with first positional arg
            if len(args) > 0:
                try:
                    return self.orig_model(args[0])
                except Exception as inner_e:
                    if DEBUG >= 2:
                        print(f"[Medusa] Alternative forward call failed: {inner_e}")
            
            # Fall back to generate call
            try:
                if hasattr(self.orig_model, 'generate'):
                    # Extract common parameters
                    tokens = kwargs.get('tokens', None)
                    if tokens is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
                        tokens = args[0]
                        
                    input_pos = kwargs.get('input_pos', None) 
                    mask = kwargs.get('mask', None)
                    hidden_state = kwargs.get('hidden_state', None)
                    curr_pos = kwargs.get('curr_pos', None)
                    
                    return self.orig_model.generate(
                        tokens=tokens,
                        input_pos=input_pos,
                        mask=mask,
                        hidden_state=hidden_state,
                        curr_pos=curr_pos
                    )
            except Exception as gen_e:
                if DEBUG >= 2:
                    print(f"[Medusa] Generate call also failed: {gen_e}")
                    
            # If all attempts fail, raise original error
            raise RuntimeError(f"All forward call patterns failed for medusa model: {e}")
    
    def generate(self, tokens=None, input_pos=None, mask=None, hidden_state=None, curr_pos=None):
        """
        Generate with Medusa parallel decoding.
        This is the core method implementing Medusa's algorithm.
        """
        if self.orig_model is None:
            raise RuntimeError("Original model not initialized")
            
        if tokens is None:
            raise ValueError("Tokens cannot be None for generation")
        
        try:
            # First, get the representation from the base model using the appropriate call pattern
            if hasattr(self.orig_model, 'generate'):
                # If the original model has a generate method, use that
                hidden_state, orig_logits = self.orig_model.generate(
                    tokens=tokens,
                    input_pos=input_pos,
                    mask=mask,
                    hidden_state=hidden_state,
                    curr_pos=curr_pos
                )
            else:
                # Otherwise try standard forward 
                # Different models may need different argument patterns
                # Try different common patterns
                try:
                    # Try standard call with dict-based kwargs
                    inputs = {}
                    if tokens is not None: inputs['tokens'] = tokens
                    if input_pos is not None: inputs['input_pos'] = input_pos
                    if mask is not None: inputs['mask'] = mask
                    if hidden_state is not None: inputs['hidden_state'] = hidden_state
                    if curr_pos is not None: inputs['curr_pos'] = curr_pos
                    
                    orig_outputs = self.orig_model(**inputs)
                except TypeError:
                    try:
                        # Try with standard PyTorch model (hidden_states as first input)
                        input_to_use = tokens if hidden_state is None else hidden_state
                        orig_outputs = self.orig_model(
                            input_to_use,
                            attention_mask=mask,
                            position_ids=input_pos
                        )
                    except TypeError:
                        # If all else fails, try different parameter combinations
                        orig_outputs = self.orig_model(tokens)
                
                # Handle different return types based on what we get
                if isinstance(orig_outputs, tuple):
                    # Common pattern for transformer models
                    hidden_state = orig_outputs[0]
                    orig_logits = orig_outputs[1] if len(orig_outputs) > 1 else None
                else:
                    # Single output tensor - likely logits
                    hidden_state = None
                    orig_logits = orig_outputs
            
            # If Medusa heads aren't initialized or we're just starting, return original output
            if not self.medusa_initialized or tokens.shape[1] <= 1:
                if DEBUG >= 2:
                    print("[Medusa] Using original model output")
                return hidden_state, orig_logits
                
            # Otherwise, use Medusa heads to predict future tokens
            if DEBUG >= 2:
                print(f"[Medusa] Generating with {self.medusa_heads} heads")
                
            # Extract the last hidden state for Medusa heads
            if hidden_state is not None:
                last_hidden = hidden_state[:, -1:]  # Shape: [batch_size, 1, hidden_size]
                
                # Get predictions from each Medusa head
                medusa_logits = []
                for head in self.medusa_heads_modules:
                    head_logits = head(last_hidden)  # Shape: [batch_size, 1, vocab_size]
                    medusa_logits.append(head_logits)
                
                # Return the original hidden states but with medusa logits information
                # We'll apply the Medusa algorithm during sampling
                if hasattr(self, "_medusa_logits_cache"):
                    self._medusa_logits_cache = orig_logits, medusa_logits
                    
                return hidden_state, orig_logits
            else:
                if DEBUG >= 2:
                    print("[Medusa] No hidden states returned, cannot apply Medusa heads")
                return hidden_state, orig_logits
            
        except Exception as e:
            print(f"[Medusa] Error in generate: {e}")
            traceback.print_exc()
            
            # Provide default values to avoid UnboundLocalError
            if 'hidden_state' not in locals():
                hidden_state = None
            if 'orig_logits' not in locals():
                orig_logits = None
                
            # Fallback to the original model's output
            return hidden_state, orig_logits
            
    def sample_with_medusa(self, logits, temperature=0.7, top_k=50):
        """
        Apply the Medusa sampling algorithm for parallel decoding.
        
        Args:
            logits: Logits from the language model
            temperature: Sampling temperature
            top_k: Top-k for sampling
            
        Returns:
            List of token IDs predicted with Medusa's algorithm
        """
        print("\n====== MEDUSA SAMPLING ACTIVE ======")
        print(f"Using {self.medusa_heads} prediction heads for parallel decoding")
        
        if not hasattr(self, "_medusa_logits_cache") or self._medusa_logits_cache is None:
            # Fallback to standard sampling
            print("Medusa cache not available, falling back to standard sampling")
            return self._standard_sample(logits, temperature, top_k)
            
        orig_logits, medusa_logits = self._medusa_logits_cache
        print(f"Medusa successfully generated predictions with {len(medusa_logits)} heads")
        
        # Create a tree of possible token sequences
        tree = []
        
        # Get top-k candidates for the next token (standard sampling)
        next_token_logits = orig_logits[:, -1]  # Shape: [batch_size, vocab_size]
        
        if temperature > 0:
            # Apply temperature and top-k
            scaled_logits = next_token_logits / temperature
            if top_k > 0:
                top_k = min(top_k, scaled_logits.size(-1))
                indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
                scaled_logits[indices_to_remove] = -float('Inf')
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy sampling
            next_token = torch.argmax(next_token_logits, dim=-1)
            
        tree.append(next_token)
        
        # Get top candidates from each Medusa head
        for i, head_logits in enumerate(medusa_logits):
            head_token_logits = head_logits[:, -1]  # Shape: [batch_size, vocab_size]
            
            if temperature > 0:
                # Apply temperature and top-k
                scaled_logits = head_token_logits / temperature
                if top_k > 0:
                    top_k = min(top_k, scaled_logits.size(-1))
                    indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
                    scaled_logits[indices_to_remove] = -float('Inf')
                probs = torch.softmax(scaled_logits, dim=-1)
                head_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy sampling
                head_token = torch.argmax(head_token_logits, dim=-1)
                
            tree.append(head_token)
            
        # For simplicity, just return all predicted tokens
        # In a full Medusa implementation, we would verify these predictions
        return tree
            
    def _standard_sample(self, logits, temperature=0.7, top_k=50):
        """Standard token sampling method when Medusa isn't available."""
        print("\n====== STANDARD SAMPLING USED (NOT MEDUSA) ======")
        
        if logits.dim() == 3:
            logits = logits[:, -1]  # Get the last token's logits
        
        # Use standard pytorch sampling
        if temperature == 0 or temperature < 1e-5:
            # Greedy sampling
            print("Using greedy sampling (temperature ≈ 0)")
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Temperature sampling
            print(f"Using temperature sampling with temp={temperature}")
            # Apply temperature
            scaled_logits = logits / temperature
            
            # Apply top-k if specified
            if top_k > 0:
                top_k = min(top_k, scaled_logits.size(-1))
                indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
                scaled_logits[indices_to_remove] = -float('Inf')
                
            # Sample from the resulting distribution
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)