import torch
from transformers import ViTConfig
from sae.saev2 import ConcatenatedInputViTSAE


# Define the configuration for the ViT model
config = ViTConfig()

# Specify the number of classes for classification
num_classes = 10  # Example: CIFAR-10 dataset

# Specify the number of early exits to be added

num_early_exits = 3

num_inputs = 2 

# Initialize the ViTSAE model
vit_sae_model = ConcatenatedInputViTSAE(config, num_classes, num_early_exits, num_inputs)

# Example input tensor (batch_size, channels, height, width)
# Assuming the input size matches what ViT expects, e.g., 3x224x224 for ImageNet
input_tensor1 = torch.rand(5, 3, 224, 224)  # Example: batch of 5 images
input_tensor2 = torch.rand(5, 3, 224, 224)


# Forward pass through the model
final_output, exit_probs = vit_sae_model([input_tensor1, input_tensor2])

# Print the final output and exit probabilities
print(final_output, exit_probs)
print("Final Output Shape:", final_output.shape)  # Should match (batch_size, num_classes)
print("Exit Probabilities:", exit_probs)
