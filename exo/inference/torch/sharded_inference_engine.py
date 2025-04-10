"""
TorchDynamicShardInferenceEngine
Sharded inference engine using PyTorch based torchtune models
"""

import os
import functools
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uuid
import re
from typing import Optional

import numpy as np
import torch
import torchtune.generation as ttg
from transformers import AutoTokenizer

from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader
from exo.inference.shard import Shard
from exo.inference.tokenizers import _resolve_tokenizer
from exo.helpers import DEBUG
from exo.inference.torch.models.llm_utils import (
  load_model_config,
  load_model_weights_torchtune,
  ShardInferenceState
)

from exo.inference.torch.models.general_mha import ShardedGeneralModel
from exo.inference.torch.medusa.medusa_model import MedusaShardedModel

# from torchtune generate recipe
# https://github.com/pytorch/torchtune/blob/main/recipes/configs/generation.yaml#L40
TEMP = 0.6
TOP_K = 35

class TorchDynamicShardInferenceEngine(InferenceEngine):
  """
  Pytorch based inferece engine for sharded models
  """
  def __init__(self, shard_downloader: ShardDownloader, medusa_enabled=False, medusa_num_heads=5, medusa_num_layers=1):
    super().__init__(shard_downloader)
    self.shard = None
    self.sharded_model = None
    self.request_id = None
    self.executor = ThreadPoolExecutor(max_workers=1)
    self.uuid = str(uuid.uuid4())
    self.model_path = None
    self.model_config = None
    self.state = None
    self.oom_cnt = 0
    self.medusa_enabled = medusa_enabled
    self.medusa_num_heads = medusa_num_heads
    self.medusa_num_layers = medusa_num_layers

    # cache settings
    self.use_cache = bool(os.getenv("TORCH_USE_CACHE", "True").lower() == "true")
    self.cache_setup = False

    # device settings
    if os.environ.get("TORCH_DEVICE"):
      self.device = torch.device(os.environ["TORCH_DEVICE"])
    elif torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

    # rng setup for sampling
    self.rng = torch.Generator(device=self.device)
    self.rng.manual_seed(1234)

  def setup_cache(self, batch_size: int=1, total_response_length: int=1024):
    # setup cache
    # this is needed for a primary node that gets the initial encoding
    if self.sharded_model is None:
      if DEBUG >= 1:
        print("[Inference] Model not initialized, skipping cache setup")
      return
        
    # Safely check if caches are enabled
    caches_enabled = False
    if hasattr(self.sharded_model, 'caches_are_enabled'):
      caches_enabled = self.sharded_model.caches_are_enabled()
    elif hasattr(self.sharded_model, 'model') and hasattr(self.sharded_model.model, 'caches_are_enabled'):
      caches_enabled = self.sharded_model.model.caches_are_enabled()
    
    if not caches_enabled and self.use_cache:
      try:
        with self.device:
          # Try different approaches to setup caches
          if hasattr(self.sharded_model, 'setup_caches'):
            self.sharded_model.setup_caches(
              batch_size=batch_size, 
              dtype=self.model_config["torch_dtype"],
              decoder_max_seq_len=total_response_length
            )
          elif hasattr(self.sharded_model, 'model') and hasattr(self.sharded_model.model, 'setup_caches'):
            self.sharded_model.model.setup_caches(
              batch_size=batch_size, 
              dtype=self.model_config["torch_dtype"],
              decoder_max_seq_len=total_response_length
            )
          
        self.cache_setup = True
      except Exception as e:
        if DEBUG >= 1:
          print(f"[Inference] Error setting up cache: {e}")

  def clear_model(self):
    """
    Clear out model and shard
    A way to avoid OOM issues
    
    All prompts are stored in VRAM
    while inference engine is up and using the same
    model class instance, this will clear it for each prompt.

    OOM issue might occur in longer chats/contexts depending on your machine.
    """
    if self.sharded_model is None or self.sharded_model.model is None:
      if DEBUG >= 1:
        print("[Inference] Model not initialized, skipping cache clear")
      return
        
    if self.sharded_model.model.caches_are_enabled():
      self.sharded_model.model.reset_caches()
    
    del self.sharded_model
    self.sharded_model = None
    
    if self.device == torch.device("cuda"):
      torch.cuda.empty_cache()
    
    self.shard = None
    self.state = None

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    if DEBUG >= 4:
      print("encode called")
      print(f"shard: {shard}")
      print(f"prompt: {prompt}")

    await self.ensure_shard(shard)

    def encode_wrapper() -> np.ndarray:
      """
      Encode the tensors from prompt along with the
      initial input_pos and mask
      """
      tokens = self.tokenizer.encode(
        prompt,
        return_tensors="pt"
      )

      # move to proper device, default is CPU
      if tokens.device != self.device:
        tokens = tokens.to(device=self.device)
      
      if DEBUG >= 4:
        print("encoded_wrapper called")
        print(f"tokens: {tokens}")

      # if going past max, just take from max onward
      if len(tokens) > self.sharded_model.max_generated_tokens:
        max_gen_tokens = self.sharded_model.max_generated_tokens
        tokens = tokens[-max_gen_tokens:]

      self.state.tokens = tokens

      bsz, tklng = tokens.size()
      total_response_length = tklng + self.sharded_model.max_generated_tokens

      self.setup_cache(bsz, total_response_length)
      
      # setup max sequence length
      max_seq_len = total_response_length
      if hasattr(self.sharded_model, 'caches_are_enabled'):
        if not self.sharded_model.caches_are_enabled():
          max_seq_len = total_response_length
        elif hasattr(self.sharded_model, 'decoder_max_cache_seq_len'):
          max_seq_len = self.sharded_model.decoder_max_cache_seq_len
      elif hasattr(self.sharded_model, 'model') and hasattr(self.sharded_model.model, 'caches_are_enabled'):
        if not self.sharded_model.model.caches_are_enabled():
          max_seq_len = total_response_length
        elif hasattr(self.sharded_model.model, 'decoder_max_cache_seq_len'):
          max_seq_len = self.sharded_model.model.decoder_max_cache_seq_len

      # set pad_id
      if hasattr(self.tokenizer, "pad_id"):
        pad_id = self.tokenizer.pad_id
      elif hasattr(self.tokenizer, "pad_token_id"):
        print(f"pad_token_id: {self.tokenizer.pad_token_id}")
        if self.tokenizer.pad_token_id is not None:
          pad_id = self.tokenizer.pad_token_id
        else:
          pad_id = 0
      else:
        pad_id = 0
      
      padding_masks = tokens != pad_id
      if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(
          padding_masks,
          (0, self.sharded_model.max_generated_tokens),
          value=True,
        )

        self.state.mask = ttg.get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_seq_len)

        self.state.input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
      else:
        self.state.mask = torch.tril(torch.ones(
          total_response_length,
          max_seq_len,
          dtype=torch.bool,
          device=self.device,
        )).unsqueeze(0)

        self.state.input_pos = torch.arange(0, total_response_length, device=self.device).unsqueeze(0)

      return tokens

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      encode_wrapper
    )

  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    if DEBUG >= 4:
      print("decode called")
      print(f"shard: {shard}")
      print(f"tokens: {tokens}")

    await self.ensure_shard(shard)

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(self.tokenizer.decode, tokens.tolist()),
    )

  async def sample(self, x: np.ndarray, temp=TEMP, top_k=TOP_K) -> np.ndarray:
    if DEBUG >= 4:
      print("sample called")
      print(f"x: {x}")
      print(f"temp: {temp}")
      print(f"top_k: {top_k}")
      print(self.device)

    logits = torch.tensor(x).to(self.device)

    def sample_wrapper():
      # Add debugging prints
      if DEBUG >= 2:
          print(f"[Sample Debug] Logits shape: {logits.shape}, dtype: {logits.dtype}, device: {logits.device}")
          print(f"[Sample Debug] Contains NaN: {torch.isnan(logits).any()}")
          print(f"[Sample Debug] Contains Inf: {torch.isinf(logits).any()}")
          print(f"[Sample Debug] temp={temp}, top_k={top_k}")
          effective_top_k = min(top_k, logits.size(-1))
          print(f"[Sample Debug] Effective top_k for torch.topk: {effective_top_k}")
          if effective_top_k <= 0:
            print("[Sample Debug] WARNING: effective_top_k is <= 0, this might cause issues!")

      # Check if we're using a Medusa model
      if self.medusa_enabled and hasattr(self.sharded_model, "sample_with_medusa"):
        try:
          print("\n[Sample Debug] *** ATTEMPTING TO USE MEDUSA SAMPLING ***")
          print(f"[Sample Debug] Medusa is enabled with {self.medusa_num_heads} heads")
          
          # Use Medusa's special sampling method
          result = self.sharded_model.sample_with_medusa(logits, temperature=temp, top_k=top_k)
          
          print("[Sample Debug] Medusa sampling completed successfully!")
          
          # For now, just return the first token
          # In a more advanced implementation, we would use all predicted tokens
          return result[0].unsqueeze(-1).cpu().numpy()
          
        except Exception as e:
          print(f"\n[Sample Debug] *** MEDUSA SAMPLING FAILED: {e} ***")
          print("[Sample Debug] Falling back to standard sampling")
      
      # Check if we're using a dummy model 
      if (self.medusa_enabled and hasattr(self.sharded_model.model, '__class__') and 
          self.sharded_model.model.__class__.__name__ == 'DummyModel'):
        if DEBUG >= 1:
          print("[Sample Debug] Using direct sampling with dummy model")
        
        # Even with zero temperature, add significant noise for variety
        # This ensures we don't just get token ID 0 repeatedly
        if temp == 0 or temp < 1e-5:
          noise = torch.randn_like(logits) * 5.0 
          tokens = torch.argmax(logits + noise, dim=-1, keepdim=True)
        else:
          # Temperature sampling with higher temperature
          effective_temp = max(temp, 0.8)  # Use at least 0.8 for dummy model
          probs = torch.nn.functional.softmax(logits / effective_temp, dim=-1)
          tokens = torch.multinomial(probs, num_samples=1)
        
        return tokens.cpu().numpy()

      # Otherwise use the normal sampling method
      q = torch.empty((logits.size(0), self.sharded_model.model.tok_embeddings.num_embeddings), device=logits.device).exponential_(1, generator=self.rng)

      tokens = ttg.sample(logits.clone(), temperature=temp, top_k=top_k, q=q.to(self.device))
      
      if DEBUG >= 4:
        print(f"tokens: {tokens}")

      return tokens.numpy(force=True)

    return await asyncio.get_running_loop().run_in_executor(self.executor, functools.partial(sample_wrapper))

  async def infer_tensor(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: Optional[dict] = None
  ) -> tuple[np.ndarray, Optional[dict]]:

    await self.ensure_shard(shard)

    # ensure shard
    if DEBUG >= 4:
      print("infer_tensor called")
      print(f"shard: {shard}")
      print(f"input_data: {input_data}")

    if inference_state.get("tokens") is not None:
      self.state.from_dict(inference_state)

    self.request_id = request_id if not self.request_id else self.request_id

    hidden_state = None
    input_tensor = None
    if input_data.ndim == 3:
      hidden_state = torch.tensor(input_data).to(
        device=self.device,
        dtype=self.model_config["torch_dtype"]
      )
    elif input_data.ndim == 2:
      input_tensor = torch.tensor(input_data).to(
        device=self.device
      )

    print(self.use_cache)
    print(self.cache_setup)

    if self.use_cache and not self.cache_setup:
      # Ensure sharded_model is initialized before accessing attributes
      if self.sharded_model is None:
        raise RuntimeError("Sharded model is not initialized. Cannot proceed with cache setup.")

      if input_tensor is not None:
        bsz, tklng = input_tensor.size()
        self.setup_cache(
          bsz,
          tklng + self.sharded_model.max_generated_tokens
        )
      else:
        print("input_tensor is None")
        # Also check state.tokens existence before accessing size
        if self.state is None or self.state.tokens is None:
          raise RuntimeError("Input tensor and state tokens are both None. Cannot determine cache size.")
        bsz, tklng = self.state.tokens.size()
        self.setup_cache(
          bsz,
          tklng + self.sharded_model.max_generated_tokens
        )

    def infer_wrapper():
      if DEBUG >= 4:
        print(f"infer_wrapper called [{self.oom_cnt} OOM]")
        print(f"self.state.tokens: {self.state.tokens}")
        print(f"hidden_state: {hidden_state}")

      # Also check sharded_model here before accessing its 'model' attribute
      if self.sharded_model is None:
        raise RuntimeError("Sharded model is not initialized inside infer_wrapper.")
        
      model_cache = self.sharded_model.model.caches_are_enabled()

      if self.state.tokens is not None:
        if input_data.ndim == 2 and input_tensor.size(-1) == 1:
          self.state.tokens = torch.cat([
            self.state.tokens.to(self.device),
            input_tensor.clone()
          ], dim=-1).to(self.device)
      else:
        self.state.tokens = input_tensor.clone()

      try:
        in_tokens = self.state.tokens.clone().to(
          device=self.device
        )

        in_input_pos = self.state.input_pos.clone().to(
          device=self.device
        )

        in_mask = self.state.mask.clone().to(
          device=self.device
        )

        if hidden_state is not None:
          model_hs, model_logits = self.sharded_model.generate(
            tokens=in_tokens,
            hidden_state=hidden_state,
            input_pos=in_input_pos,
            mask=in_mask,
            curr_pos=self.state.curr_pos
          )
        else:
          if not model_cache:
            model_hs, model_logits = self.sharded_model.generate(
              tokens=in_tokens,
              input_pos=in_input_pos,
              mask=in_mask,
              curr_pos=self.state.curr_pos
            )
          else:
            model_hs, model_logits = self.sharded_model.generate(
              tokens=input_tensor,
              input_pos=in_input_pos,
              mask=in_mask,
              curr_pos=self.state.curr_pos
            )
      except torch.cuda.OutOfMemoryError:
        print(f"OOM on cuda, clearing model and stopping")
        self.oom_cnt += 1
        self.clear_model()
        return
      except Exception as err:
        print(f"infer_tensor err\n{err}")
        raise

      if model_hs is not None:
        # numpy current no support for bf16
        if model_hs.dtype == torch.bfloat16:
          model_hs = model_hs.float()

        if DEBUG >= 4:
          print("sending hidden states")
          print(f"model_hs: {model_hs.size()}")
          print(f"state.tokens: {self.state.tokens}")
          print(f"state.input_pos: {self.state.input_pos.size()}")
          print(f"state.mask: {self.state.mask.size()}")
        
        return (
          model_hs.numpy(force=True),
          self.state.to_dict(),
        )
      
      if self.state.curr_pos == 0:
        self.state.curr_pos = self.state.tokens.size(-1)
      else:
        self.state.curr_pos += 1

      # numpy current no support for bf16
      if model_logits.dtype == torch.bfloat16:
        model_logits = model_logits.float()

      return (
        model_logits[:, -1].numpy(force=True),
        self.state.to_dict(),
      )

    return await asyncio.get_running_loop().run_in_executor(self.executor, infer_wrapper)

  async def ensure_shard(self, shard: Shard):
    if DEBUG >= 4:
      print("shard ensured\n")
      print(f"shard: {shard}")
      print(f"class shard: {self.shard}")
      print(f"uuid: {self.uuid}")

    # reset model after last layer to fix OOM
    if self.shard == shard:
      return

    self.shard = shard

    # Using CPU to store inference state
    self.state = ShardInferenceState()

    # download model safetensors and shard

    self.model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
    self.model_config = load_model_config(self.model_path/"config.json")

    # self.tokenizer = await _resolve_tokenizer(model_path)
    self.tokenizer = await _resolve_tokenizer(self.model_path)

    def start_model_sync(): # Renamed for clarity
      local_sharded_model = None # Use a local variable
      try:
        if DEBUG >= 4:
          print("start_model_sync called")

        if self.medusa_enabled:
          if DEBUG >= 1:
            print(f"[Medusa] Creating Medusa-enhanced model with {self.medusa_num_heads} heads and {self.medusa_num_layers} layers")
          
          try:
            # First create the regular model
            base_model = ShardedGeneralModel(
              config=self.model_config,
              shard=shard,
              device=self.device,
              dtype=self.model_config["torch_dtype"],
              use_cache=self.use_cache
            )
            
            # Load weights for the base model
            load_model_weights_torchtune(
              cache_dir=self.model_path,
              shard=shard,
              model=base_model,
              num_heads=self.model_config["num_heads"],
              num_kv_heads=self.model_config["num_kv_heads"],
              dim=self.model_config["embed_dim"],
              head_dim=self.model_config["head_dim"]
            )
            
            # Create Medusa-enhanced model
            local_sharded_model = MedusaShardedModel(
              config=self.model_config,
              shard=shard,
              device=self.device,
              dtype=self.model_config["torch_dtype"],
              use_cache=self.use_cache,
              medusa_num_heads=self.medusa_num_heads,
              medusa_num_layers=self.medusa_num_layers
            )
            
            # Initialize with the base model
            local_sharded_model.initialize_original_model(base_model)
            
            if DEBUG >= 1:
              print(f"[Medusa] Successfully created Medusa-enhanced model")
              
          except Exception as e:
            print(f"[Medusa] Error creating Medusa model: {e}")
            print(f"[Medusa] Falling back to standard model")
            # Fall back to standard model if Medusa fails
            local_sharded_model = ShardedGeneralModel(
              config=self.model_config,
              shard=shard,
              device=self.device,
              dtype=self.model_config["torch_dtype"],
              use_cache=self.use_cache
            )
            load_model_weights_torchtune(
              cache_dir=self.model_path,
              shard=shard,
              model=local_sharded_model,
              num_heads=self.model_config["num_heads"],
              num_kv_heads=self.model_config["num_kv_heads"],
              dim=self.model_config["embed_dim"],
              head_dim=self.model_config["head_dim"]
            )
        else:
          local_sharded_model = ShardedGeneralModel(
            config=self.model_config,
            shard=shard,
            device=self.device,
            dtype=self.model_config["torch_dtype"],
            use_cache=self.use_cache
          )
          load_model_weights_torchtune(
            cache_dir=self.model_path,
            shard=shard,
            model=local_sharded_model,
            num_heads=self.model_config["num_heads"],
            num_kv_heads=self.model_config["num_kv_heads"],
            dim=self.model_config["embed_dim"],
            head_dim=self.model_config["head_dim"]
          )

        if local_sharded_model is None:
            print("[start_model_sync Error] ShardedModel creation returned None!")
            return None # Return None on failure

        print("After sharded_model creation..............")

        if DEBUG >= 1:
            print(f"[start_model_sync Success] Successfully initialized and loaded weights for shard: {shard}")
        return local_sharded_model # Return the initialized model
            
      except Exception as e:
        print(f"[start_model_sync Error] Failed to initialize or load weights for shard {shard}: {e}")
        import traceback
        traceback.print_exc()
        return None # Return None on failure
    
    # Execute start_model_sync in the executor
    if DEBUG >= 2:
      print(f"[ensure_shard] Running start_model_sync in executor for shard: {shard}")
    # Use run_in_executor correctly to get the return value
    initialized_model = await asyncio.get_running_loop().run_in_executor(
      self.executor,
      start_model_sync # Pass the function directly
    )
    
    # Assign the result to the instance variable AFTER awaiting
    self.sharded_model = initialized_model 
    if self.sharded_model is not None:
        if DEBUG >= 1:
            print(f"[ensure_shard] Successfully assigned initialized model for shard: {self.shard}")
    else:
        print(f"[ensure_shard] Failed to initialize model for shard: {self.shard}. self.sharded_model is None.")

  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
