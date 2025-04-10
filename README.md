# Medusa Integration for Exo Framework

A comprehensive implementation of Medusa parallel decoding to accelerate inference in the Exo distributed AI framework.

## Overview
This project integrates Medusa, a speculative decoding technique for accelerating LLM inference, with the Exo framework. The architecture aims to enable parallel token generation for faster inference.

## Current Status
The implementation is a work in progress. Debug logs show Medusa is initializing but falling back to standard sampling due to cache availability issues. Various errors are being encountered, including:
- Cache setup problems
- Parameter mismatches in function calls
- Model compatibility issues

## Key Modified Components
- MedusaShardedModel (exo/inference/torch/medusa/medusa_model.py)
- ShardedGeneralModel (exo/inference/torch/models/general_mha.py)
- ShardTransformerDecoder (exo/inference/torch/models/llm_utils.py)
- TorchDynamicShardInferenceEngine (exo/inference/torch/sharded_inference_engine.py)

## Usage
Run the model with Medusa enabled:
```bash
./run_exo_medusa.sh --prompt "Your prompt here"
```

## Implementation Details
Medusa introduces parallel decoding by generating multiple tokens in a single forward pass. This implementation adds the necessary infrastructure to the Exo framework to support this capability, although it currently falls back to standard sampling due to cache handling issues.

## How Medusa Works
Medusa uses a tree-like structure to predict multiple possible future tokens simultaneously. The approach:
1. Generates the next token conventionally
2. Uses additional prediction heads to generate potential future tokens
3. Verifies these speculative tokens with the main model
4. Accepts multiple tokens at once if they match predictions


This parallel prediction can significantly speed up inference by reducing the number of forward passes needed.


# Medusa Integration for Exo Framework

A comprehensive implementation of Medusa parallel decoding to accelerate inference in the Exo distributed AI framework.

## Overview
This project integrates Medusa, a speculative decoding technique for accelerating LLM inference, with the Exo framework. Medusa is designed to enable parallel token generation through multiple prediction heads, which can potentially speed up inference compared to sequential token generation.

## Current Status
The implementation is in progress with several challenges:
- Debug logs show Medusa is initializing but consistently falling back to standard sampling due to cache availability issues
- Various errors are being encountered including:
  - Cache setup problems (`ShardedGeneralModel object has no attribute 'caches_are_enabled'`)
  - Parameter mismatches in function calls (`ShardTransformerDecoder.setup_caches() takes 3 positional arguments but 4 were given`)
  - Model compatibility issues (`TypeError: _forward_unimplemented() got an unexpected keyword argument 'tokens'`)
  - Missing attributes (`'ShardedGeneralModel' object has no attribute 'decoder_max_cache_seq_len'`)

## Key Modified Components
- **MedusaShardedModel** (`exo/inference/torch/medusa/medusa_model.py`)
  - Core implementation that wraps the original model with Medusa functionality
  - Manages multiple prediction heads for parallel token generation
  - Handles fallback to standard sampling when needed

- **ShardedGeneralModel** (`exo/inference/torch/models/general_mha.py`)
  - Modified to support Medusa's cache requirements
  - Includes modified `setup_caches` method that's causing parameter mismatch errors

- **ShardTransformerDecoder** (`exo/inference/torch/models/llm_utils.py`)
  - Adapted to support Medusa's specialized caching requirements
  - Parameter handling for setup_caches function causing issues

- **TorchDynamicShardInferenceEngine** (`exo/inference/torch/sharded_inference_engine.py`)
  - Contains the token sampling logic that attempts to use Medusa
  - Includes debug output to track Medusa operation
  - Implements fallback mechanism when Medusa cache is unavailable

## Usage
Run the model with Medusa enabled:
```bash
cd /home/siu856580840/exo/VANILLA/exo\ vanilla/exo-pt-main
./run_exo_medusa.sh --prompt "Your prompt here"
```

Additional options:
```bash
./run_exo_medusa.sh --model "qwen-2.5-0.5b" --prompt "Your prompt here" --heads 8 --layers 1
```

The script automatically activates the "exo-vanilla" virtual environment located at `/home/siu856580840/exo/VANILLA/exo vanilla/exo-pt-main/exo-vanilla`.

## How Medusa Works

Medusa accelerates LLM inference through a technique called speculative decoding:

1. **Basic Concept**: Traditional LLMs generate text one token at a time sequentially. Medusa attempts to predict multiple future tokens in parallel, verifying them against the main model's output.

2. **Architecture**:
   - A base model (e.g., Qwen) generates the first token
   - Multiple "Medusa heads" (specialized prediction layers) predict potential future tokens
   - These predictions form a tree-like structure of possible continuations
   - The main model verifies these speculative predictions
   - When predictions match, multiple tokens can be accepted in one step

3. **Token Generation Process**:
   ```
   [Base Model Token] → [Medusa Head 1] → [Prediction 1.1]
                                        → [Prediction 1.2]
                      → [Medusa Head 2] → [Prediction 2.1]
                                        → [Prediction 2.2]
   ```

4. **Performance Advantages**: When working correctly, Medusa can significantly reduce the number of forward passes needed, accelerating text generation without sacrificing quality.

## Implementation Details

### Core MedusaShardedModel Class
```python
class MedusaShardedModel(nn.Module):
    def __init__(self, orig_model, num_heads=8, num_layers=1):
        super().__init__()
        self.orig_model = orig_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        # Initialize Medusa prediction heads
        self.medusa_heads = nn.ModuleList([
            nn.Linear(self.orig_model.config.hidden_size, self.orig_model.config.vocab_size)
            for _ in range(num_heads * num_layers)
        ])
        print("After sharded_model creation..............")
        
    def generate(self, tokens, **kwargs):
        try:
            # This line causes the error seen in logs:
            # TypeError: _forward_unimplemented() got an unexpected keyword argument 'tokens'
            orig_outputs = self.orig_model.forward(
                tokens=tokens,
                **kwargs
            )
            # Rest of token generation logic...
        except Exception as e:
            print(f"[Medusa] Error in generate: {e}")
            import traceback
            traceback.print_exc()
            # Fallback path
            return hidden_state, orig_logits  # Causes "cannot access local variable" error
```

### Sampling Logic in Inference Engine
```python
def sample_token(self, logits, prev_tokens, inference_state):
    try:
        print("[Sample Debug] *** ATTEMPTING TO USE MEDUSA SAMPLING ***")
        print(f"[Sample Debug] Medusa is enabled with {self.medusa_num_heads} heads")
        
        print("====== MEDUSA SAMPLING ACTIVE ======")
        print(f"Using {self.medusa_num_heads} prediction heads for parallel decoding")
        
        # Critical check that's failing
        if not hasattr(self.sharded_model, 'medusa_cache') or self.sharded_model.medusa_cache is None:
            print("Medusa cache not available, falling back to standard sampling")
            print("====== STANDARD SAMPLING USED (NOT MEDUSA) ======")
            
            # Fallback to standard token-by-token sampling
            if self.temperature < 0.01:
                print("Using greedy sampling (temperature ≈ 0)")
                next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            else:
                print(f"Using temperature sampling (temp={self.temperature})")
                probs = torch.softmax(logits[:, -1] / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            return next_token, inference_state
    except Exception as e:
        print(f"Error in Medusa sampling: {e}")
        # Additional fallback logic...
```

### Cache Setup (Where Errors Occur)
```python
def setup_cache(self, batch_size, max_seq_len):
    # Causes 'ShardedGeneralModel' object has no attribute 'caches_are_enabled' error
    if not self.sharded_model.model.caches_are_enabled() and self.use_cache:
        print("Setting up caches...")
        
        # This causes "ShardTransformerDecoder.setup_caches() takes 3 positional arguments but 4 were given" error
        self.sharded_model.model.setup_caches(
            batch_size,
            torch.float16,  # dtype
            max_seq_len
        )
        
        # Medusa cache initialization attempt
        if hasattr(self.sharded_model, 'medusa_heads') and self.medusa_enabled:
            self.sharded_model.medusa_cache = {
                'key_values': None,
                'predictions': []
            }
            print("Medusa cache initialized!")
```

### Terminal Output
When running Medusa, the terminal output shows:
```
[Sample Debug] *** ATTEMPTING TO USE MEDUSA SAMPLING ***
[Sample Debug] Medusa is enabled with 8 heads
====== MEDUSA SAMPLING ACTIVE ======
Using 8 prediction heads for parallel decoding
Medusa cache not available, falling back to standard sampling
====== STANDARD SAMPLING USED (NOT MEDUSA) ======
Using greedy sampling (temperature ≈ 0)
[Sample Debug] Medusa sampling completed successfully!
```

This demonstrates that while Medusa is correctly initialized with the specified number of heads, it consistently falls back to standard sampling due to cache initialization issues.

## Known Issues

1. **Cache Initialization Failure**: The most critical issue is that the Medusa cache is not being properly initialized, causing fallback to standard sampling.

2. **Model Compatibility**: The original model's forward method does not accept the parameters being passed by the Medusa wrapper.

3. **Parameter Mismatches**: Several function calls have parameter count mismatches, including the `setup_caches()` method.

4. **Missing Attributes**: The implementation attempts to access attributes that don't exist on the model classes.

## Future Work

1. **Fix Cache Initialization**: Properly initialize the Medusa cache to enable parallel decoding.

2. **Address Model Compatibility**: Adapt the Medusa implementation to match the expected parameters of the underlying model.

3. **Standardize Parameter Interfaces**: Ensure consistent parameter counts and types across all function calls.

4. **Implement Better Error Handling**: Provide more graceful fallbacks and clearer error messages.

5. **Optimize Performance**: Once basic functionality is working, optimize the implementation for maximum speed improvement.

## Future Work
- Fix cache initialization issues
- Properly integrate with different model architectures
- Optimize performance for various head configurations
- Improve error handling and debug output
