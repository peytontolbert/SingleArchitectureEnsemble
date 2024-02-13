import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class EarlyExitLayer(nn.Module):
    """Early exit layer for SAE, placed at certain depths in the ViT model."""
    def __init__(self, hidden_size, num_classes):
        super(EarlyExitLayer, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            # Optional: Add more layers as needed for the early exit
        )

    def forward(self, x):
        return self.classifier(x)

class ViTSAE(nn.Module):
    def __init__(self, config, num_classes, num_early_exits):
        super(ViTSAE, self).__init__()
        self.vit = ViTModel(config)
        self.num_classes = num_classes
        self.num_early_exits = num_early_exits

        # Define early exits based on the number of early exits
        self.early_exits = nn.ModuleList([
            EarlyExitLayer(config.hidden_size, num_classes) for _ in range(num_early_exits)
        ])

        # Variational parameters θ for each exit, initialized as logits
        self.exit_logits = nn.Parameter(torch.zeros(num_early_exits), requires_grad=True)
        self.temperature = 1.0  # Temperature for softmax scaling, can be adjusted

    def forward(self, x, temperature=None):
        if temperature is None:
            temperature = self.temperature

        outputs = []
        exit_probs = F.softmax(self.exit_logits / temperature, dim=0)

        control_flow = x
        hidden_states = self.vit(control_flow).last_hidden_state

        for i, exit_layer in enumerate(self.early_exits):
            exit_input = hidden_states[:, 0]  # Use the [CLS] token's representation
            exit_output = exit_layer(exit_input)
            outputs.append(exit_output)

        # Combine outputs based on exit probabilities
        combined_output = torch.stack([exit_probs[i] * outputs[i] for i in range(self.num_early_exits)])
        final_output = combined_output.sum(dim=0)

        return final_output, exit_probs

class ConcatenatedInputViTSAE(ViTSAE):
    def __init__(self, config, num_classes, num_early_exits, num_inputs):
        super().__init__(config, num_classes, num_early_exits)
        self.num_inputs = num_inputs

    def forward(self, inputs, temperature=None):
        # Adjusted method to handle inputs without increasing channel dimension incorrectly
        if temperature is None:
            temperature = self.temperature

        # Process each input separately and collect outputs
        outputs = []
        exit_probs = F.softmax(self.exit_logits / temperature, dim=0)

        for input_tensor in inputs:
            
            # print(f"input tensor before unsqueeze {input_tensor.shape}")

            input_tensor = input_tensor.unsqueeze(0)

            # print(f"input tensor after unsqueeze {input_tensor.shape}")

            hidden_states = self.vit(input_tensor).last_hidden_state
            for i, exit_layer in enumerate(self.early_exits):
                if len(outputs) <= i:
                    outputs.append([])
                exit_input = hidden_states[:, 0]  # Use the [CLS] token's representation
                outputs[i].append(exit_layer(exit_input))

        # Aggregate outputs from each input
        aggregated_outputs = []
        for output_group in outputs:
            # Example aggregation method: averaging outputs from the same exit across all inputs
            aggregated_outputs.append(torch.stack(output_group).mean(dim=0))

        combined_output = torch.stack([exit_probs[i] * aggregated_outputs[i] for i in range(self.num_early_exits)])
        final_output = combined_output.sum(dim=0)

        return final_output, exit_probs
