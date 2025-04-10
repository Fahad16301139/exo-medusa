"""
Example script to run Medusa model with EXO framework.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import medusa
        import transformers
        import torch
        logger.info("All required dependencies are installed.")
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        logger.error("Please install the required dependencies:")
        logger.error("1. python -m pip install --upgrade setuptools")
        logger.error("2. python -m pip install medusa-llm --no-deps")
        logger.error("3. python -m pip install transformers torch")
        sys.exit(1)

async def main():
    try:
        # Check dependencies first
        check_dependencies()
        
        # Import after dependency check to avoid circular imports
        from exo.download.new_shard_download import new_shard_downloader
        from exo.inference.torch.medusa import MedusaInferenceEngine
        from exo.inference.shard import Shard
        from exo.helpers import DEBUG
        
        # Set environment variables
        os.environ["TORCH_USE_CACHE"] = "true"
        if "DEBUG" not in os.environ:
            os.environ["DEBUG"] = "2"
            
        logger.info("Initializing shard downloader...")
        shard_downloader = new_shard_downloader()
        
        logger.info("Creating Medusa inference engine...")
        medusa_engine = MedusaInferenceEngine(
            shard_downloader=shard_downloader,
            medusa_num_heads=5,
            medusa_num_layers=1
        )
        
        logger.info("Setting up shard configuration...")
        shard = Shard(
            model_id="llama-2-7b",
            start_layer=0,
            end_layer=31,
            n_layers=32
        )
        
        logger.info("Downloading and initializing shard...")
        await medusa_engine.ensure_shard(shard)
        
        prompt = "Hello, how are you today?"
        logger.info(f"Generating response for prompt: {prompt}")
        
        response = await medusa_engine.generate(
            prompt,
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )
        
        logger.info("Generation complete!")
        print("\nPrompt:", prompt)
        print("Response:", response)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if DEBUG >= 1:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 