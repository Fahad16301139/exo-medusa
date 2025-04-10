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

# Command options:
# --model: Model name (default: qwen-2.5-0.5b)
# --prompt: Prompt for the model (default: "Who are you?")
# --heads: Number of Medusa heads (default: 8)
# --layers: Number of Medusa layers (default: 1) 