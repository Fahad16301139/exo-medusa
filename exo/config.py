def _get_default_inference_engine():
  """
  Returns the default inference engine based on the platform.
  """
  if is_mac():
    if is_mlx_available():
      return "MLXDynamicShardInferenceEngine"
    elif is_tinygrad_available():
      return "TinyGradDynamicShardInferenceEngine"
  else:
    if is_torch_available():
      return "TorchDynamicShardInferenceEngine"
    elif is_tinygrad_available():
      return "TinyGradDynamicShardInferenceEngine"
  return None 