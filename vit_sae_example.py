import torch
from transformers import ViTConfig
from sae.saev2 import ConcatenatedInputViTSAE

# Initialize model with specified configuration, class count, and early exit count
config = ViTConfig()
vit_sae_model = ConcatenatedInputViTSAE(config, 
                                        num_classes=10, 
                                        num_early_exits=3, 
                                    )

# Prepare example input tensors for the model
input_tensor1 = torch.rand(5, 3, 224, 224)
input_tensor2 = torch.rand(5, 3, 224, 224)
concatenated_input = torch.cat((input_tensor1, input_tensor2), dim=0)

# Execute a forward pass and print the outcomes
final_output, exit_probs = vit_sae_model(concatenated_input)
print(final_output, exit_probs)
print("Final Output Shape:", final_output.shape)
print("Exit Probabilities:", exit_probs)
