# Medusa Integration for Exo Framework

A comprehensive implementation of Medusa parallel decoding to accelerate inference in the Exo distributed AI framework.

## Overview
This project integrates Medusa, a speculative decoding technique for accelerating LLM inference, with the Exo framework. Medusa is designed to enable parallel token generation through multiple prediction heads, which can potentially speed up inference compared to sequential token generation.

## What is Medusa and How It Works

Medusa is a specialized technique for making large language models generate text faster. Traditional LLMs generate one token (word piece) at a time, which is slow. Medusa uses a clever approach called "speculative decoding" to predict multiple future tokens simultaneously:

1. **Basic Concept**: Instead of generating just one token at a time, Medusa tries to "look ahead" and predict several tokens that might come next in the sequence.

2. **Prediction Heads**: Medusa adds special "prediction heads" on top of the base model. These are additional neural network layers that predict what future tokens might be.

3. **Tree Structure**: The predictions form a tree-like structure of possibilities. For example, after generating "The cat", Medusa might predict {"sat", "was", "jumped"} as possible next tokens, and for each of those, it predicts additional tokens.

4. **Verification Process**: The main LLM verifies these speculative predictions. If they match what the main model would have produced, multiple tokens can be accepted at once, saving time.

5. **Speedup Potential**: When working correctly, Medusa can generate text 2-3x faster by reducing the number of forward passes needed through the neural network.

## How We Implemented Medusa in Exo Framework

The Exo framework is a distributed system for running large language models. To integrate Medusa into this system, we needed to make several key modifications:

### 1. MedusaShardedModel Class (Core Implementation)

This is the heart of the Medusa implementation. We created a new class that wraps the original model and adds the prediction heads:

```python
class MedusaShardedModel(nn.Module):
    def __init__(self, orig_model, num_heads=8, num_layers=1):
        super().__init__()
        self.orig_model = orig_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        # Initialize Medusa prediction heads - these are the additional
        # layers that predict future tokens
        self.medusa_heads = nn.ModuleList([
            nn.Linear(self.orig_model.config.hidden_size, self.orig_model.config.vocab_size)
            for _ in range(num_heads * num_layers)
        ])
        print("After sharded_model creation..............")
        
    def generate(self, tokens, **kwargs):
        try:
            # Forward pass through the original model
            # This is where we're getting the error:
            # TypeError: _forward_unimplemented() got an unexpected keyword argument 'tokens'
            orig_outputs = self.orig_model.forward(
                tokens=tokens,
                **kwargs
            )
            
            # Get hidden states and logits from original model output
            orig_hidden_state = orig_outputs[0]  # hidden state
            orig_logits = orig_outputs[1]  # logits
            
            # Here's where the Medusa magic happens - using the prediction heads
            # to generate multiple possible future tokens
            if hasattr(self, 'medusa_heads') and self.medusa_heads:
                medusa_logits = []
                for head in self.medusa_heads:
                    # Each head predicts potential future tokens
                    head_logits = head(orig_hidden_state[:, -1:])
                    medusa_logits.append(head_logits)
                
                # Return both original model output and Medusa predictions
                return orig_hidden_state, (orig_logits, medusa_logits)
            else:
                return orig_hidden_state, orig_logits
                
        except Exception as e:
            # Error handling with detailed traceback
            print(f"[Medusa] Error in generate: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to standard output if there's an error
            # This is reaching this point in our implementation
            return hidden_state, orig_logits  # This causes "cannot access local variable" error
```

This class introduces the Medusa architecture to the Exo framework. It creates specialized prediction heads (simple linear layers) that take the model's hidden states and predict potential future tokens. The number of heads is configurable (default: 8), and each head focuses on predicting different possible continuations.

### 2. Sampling Logic in Inference Engine

We modified the token sampling process in the Exo framework to handle Medusa's parallel predictions. This code determines whether to use Medusa sampling or fall back to standard sampling:

```python
def sample_token(self, logits, prev_tokens, inference_state):
    try:
        # Debug output showing Medusa initialization
        print("[Sample Debug] *** ATTEMPTING TO USE MEDUSA SAMPLING ***")
        print(f"[Sample Debug] Medusa is enabled with {self.medusa_num_heads} heads")
        
        print("====== MEDUSA SAMPLING ACTIVE ======")
        print(f"Using {self.medusa_num_heads} prediction heads for parallel decoding")
        
        # Critical check: Is the Medusa cache available?
        # This is where our implementation is failing - the cache is never properly initialized
        if not hasattr(self.sharded_model, 'medusa_cache') or self.sharded_model.medusa_cache is None:
            print("Medusa cache not available, falling back to standard sampling")
            print("====== STANDARD SAMPLING USED (NOT MEDUSA) ======")
            
            # Fall back to standard token-by-token sampling when Medusa cache isn't available
            if self.temperature < 0.01:
                print("Using greedy sampling (temperature ≈ 0)")
                next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            else:
                print(f"Using temperature sampling (temp={self.temperature})")
                probs = torch.softmax(logits[:, -1] / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            print("[Sample Debug] Medusa sampling completed successfully!")
            return next_token, inference_state
            
        else:
            # This section would handle actual Medusa parallel prediction
            # but we never reach this code because the cache is never properly initialized
            print("Using Medusa tree-based sampling")
            # Get predictions from Medusa heads
            medusa_logits = self.sharded_model.medusa_logits
            
            # Process tree of predictions to find best path
            # ... (tree verification logic would go here)
            
            # Accept multiple tokens if verified
            return accepted_tokens, inference_state
            
    except Exception as e:
        print(f"Error in Medusa sampling: {e}")
        # Fall back to standard sampling on error
        # ... (standard sampling fallback)
        return next_token, inference_state
```

This sampling function demonstrates how Medusa should work when properly implemented. It first checks if the Medusa cache is available. In our current implementation, this check is failing ("Medusa cache not available"), causing it to fall back to standard sampling. The logs consistently show this fallback happening.

### 3. Cache Setup - Where the Problems Occur

The most problematic part of our implementation is the cache setup. This is where most of the errors are occurring:

```python
def setup_cache(self, batch_size, max_seq_len):
    # This line causes one of our key errors:
    # 'ShardedGeneralModel' object has no attribute 'caches_are_enabled'
    if not self.sharded_model.model.caches_are_enabled() and self.use_cache:
        print("Setting up caches...")
        
        # This causes another key error:
        # "ShardTransformerDecoder.setup_caches() takes 3 positional arguments but 4 were given"
        self.sharded_model.model.setup_caches(
            batch_size,
            torch.float16,  # dtype
            max_seq_len
        )
        
        # This is where we try to initialize the Medusa cache
        # but we never reach this code due to the previous errors
        if hasattr(self.sharded_model, 'medusa_heads') and self.medusa_enabled:
            self.sharded_model.medusa_cache = {
                'key_values': None,
                'predictions': []
            }
            print("Medusa cache initialized!")
```

This function attempts to set up the caching system that Medusa needs to work. However, it's encountering several errors:
1. The `caches_are_enabled()` method doesn't exist on the model class
2. The `setup_caches()` method has a parameter mismatch
3. Because of these errors, the Medusa cache is never properly initialized

### 4. Script to Run Medusa

We created a script called `run_exo_medusa.sh` to make it easy to run Medusa with different options:

```bash
#!/bin/bash

# Activate the virtual environment
source $(dirname "$0")/exo-vanilla/Scripts/activate

# Default model parameters
MODEL="qwen-2.5-0.5b"
PROMPT="Who are you?"
MEDUSA_HEADS=8
MEDUSA_LAYERS=1

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --heads)
      MEDUSA_HEADS="$2"
      shift 2
      ;;
    --layers)
      MEDUSA_LAYERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Run exo with Medusa enabled
python -m exo.main run "$MODEL" --prompt "$PROMPT" --medusa-enabled --medusa-num-heads "$MEDUSA_HEADS" --medusa-num-layers "$MEDUSA_LAYERS" --inference-engine torch
```

This script provides a convenient interface to run the Exo framework with Medusa enabled. It activates the virtual environment, sets default values for the model, prompt, and Medusa parameters, and then parses any command-line arguments to override these defaults.

### 5. Main Entry Point for Medusa

We modified the main application file to initialize Medusa when the `--medusa-enabled` flag is passed:

```python
def run_model_cli():
    args = parse_args()
    
    # Initialize the model
    model = initialize_model(args.model_name)
    
    # If Medusa is enabled, wrap the model with our MedusaShardedModel
    if args.medusa_enabled:
        from exo.inference.torch.medusa.medusa_model import MedusaShardedModel
        model = MedusaShardedModel(model, num_heads=args.medusa_num_heads, num_layers=args.medusa_num_layers)
        print(f"Medusa enabled with {args.medusa_num_heads} heads and {args.medusa_num_layers} layers")
```

This code initializes the base model and then, if Medusa is enabled, wraps it with our `MedusaShardedModel` class. This is the entry point for the entire Medusa functionality.

## Current Status and Error Logs

Looking at the logs we can see that Medusa is being correctly initialized with the specified number of heads, but several key errors prevent it from working properly:

```
After sharded_model creation..............
[Sample Debug] *** ATTEMPTING TO USE MEDUSA SAMPLING ***
[Sample Debug] Medusa is enabled with 8 heads
====== MEDUSA SAMPLING ACTIVE ======
Using 8 prediction heads for parallel decoding
Medusa cache not available, falling back to standard sampling
====== STANDARD SAMPLING USED (NOT MEDUSA) ======
Using greedy sampling (temperature ≈ 0)
[Sample Debug] Medusa sampling completed successfully!
```

The key issues are:

1. **Cache Initialization Failure**: The most critical issue is that the Medusa cache is not being properly initialized, causing it to fall back to standard sampling.

2. **Parameter and Type Mismatches**: We're encountering errors like:
   - `TypeError: _forward_unimplemented() got an unexpected keyword argument 'tokens'`
   - `ShardTransformerDecoder.setup_caches() takes 3 positional arguments but 4 were given`
   - `'ShardedGeneralModel' object has no attribute 'caches_are_enabled'`

3. **Missing Attributes**: The code tries to access attributes that don't exist on the model objects, such as `decoder_max_cache_seq_len`.

Despite these issues, the model is still functioning - but it's falling back to standard token-by-token generation rather than using Medusa's parallel prediction. This means we've set up the infrastructure for Medusa, but the core parallel speedup isn't being achieved yet.

## Known Issues and Future Work

1. **Fix Cache Initialization**: The most pressing issue is to properly initialize the Medusa cache to enable parallel decoding.

2. **Address Model Compatibility**: We need to adapt the Medusa implementation to match the expected parameters of the underlying model.

3. **Standardize Parameter Interfaces**: Ensure consistent parameter counts and types across all function calls.

4. **Implement Better Error Handling**: Provide more graceful fallbacks and clearer error messages.

5. **Optimize Performance**: Once basic functionality is working, optimize the implementation for maximum speed improvement.
