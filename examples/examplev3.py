from transformers import ViTConfig
from SAEv3 import ViTSAE
import torch

# Example usage:
config = ViTConfig()
num_classes = 10  # example number of classes
num_early_exits = 3  # example number of early exits
exit_thresholds = [0.8, 0.9, 0.95]  # example thresholds for each early exit layer

# Initialize the model
vit_sae_model = ViTSAE(config, num_classes, num_early_exits, exit_thresholds)

# Example input
x = torch.rand((1, 3, 224, 224))  # example batch of input images

# Forward pass, calculate early exits
output, early_exit_index = vit_sae_model(x)
print(output)
