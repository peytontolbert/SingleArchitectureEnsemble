from SAE import ViTSAE
from transformers import ViTConfig
import torch

# Example usage
config = ViTConfig(image_size=224, num_hidden_layers=12, hidden_size=768, num_attention_heads=12, num_labels=10)
model = ViTSAE(config, num_classes=10, early_exit_thresholds=[0.9, 0.95, 0.99])

# Example input tensor (batch size, channels, height, width)
x = torch.randn(2, 3, 224, 224)
output, exit_point = model(x)
print(f"Output shape: {output.shape}, Exit point: {exit_point}")