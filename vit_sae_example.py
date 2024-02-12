
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from sae.saev2 import ConcatenatedInputViTSAE

# Example usage:
# Initialize the ViT configuration
vit_config = ViTConfig()

# Specify the number of classes for classification
num_classes = 10

# Specify the number of early exits
num_early_exits = 3

# Specify the number of inputs to concatenate
num_inputs = 2

# Create the model
model = ConcatenatedInputViTSAE(vit_config, num_classes, num_early_exits, num_inputs)

# Dummy inputs for testing
input1 = torch.randn(1, 256, 768)  # Example input for one image
input2 = torch.randn(1, 256, 768)  # Example input for another image

# Forward pass with concatenated inputs
outputs, exit_probs = model([input1, input2])
