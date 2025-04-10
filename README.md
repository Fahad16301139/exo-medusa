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

## Future Work
- Fix cache initialization issues
- Properly integrate with different model architectures
- Optimize performance for various head configurations
- Improve error handling and debug output
