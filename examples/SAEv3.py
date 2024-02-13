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
    def __init__(self, config, num_classes, num_early_exits, exit_thresholds):
        super(ViTSAE, self).__init__()
        self.vit = ViTModel(config)
        self.num_classes = num_classes
        self.num_early_exits = num_early_exits
        self.exit_thresholds = exit_thresholds  # Example thresholds; can be adjusted
        # Define early exits based on the number of early exits
        self.early_exits = nn.ModuleList(
            [
                EarlyExitLayer(config.hidden_size, num_classes)
                for _ in range(num_early_exits)
            ]
        )

        # Variational parameters θ for each exit, initialized as logits
        self.exit_logits = nn.Parameter(
            torch.zeros(num_early_exits), requires_grad=True
        )
        self.temperature = 1.0  # Temperature for softmax scaling, can be adjusted

    def forward(self, x):
        control_flow = x
        hidden_states = self.vit(control_flow).last_hidden_state
        confidences = []

        for i, exit_layer in enumerate(self.early_exits):
            exit_input = hidden_states[:, 0]  # Use the [CLS] token's representation
            exit_output = exit_layer(exit_input)
            exit_confidence = torch.max(F.softmax(exit_output, dim=1), dim=1)[0]
            confidences.append(exit_confidence)
            if exit_confidence > self.exit_thresholds[i]:
                # Take the early exit if confidence exceeds the threshold
                return exit_output, i  # Returning the output and index of early exit

        # All exits were below the threshold; return the final layer's output and index
        return exit_output, len(self.early_exits)


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

        combined_output = torch.stack(
            [exit_probs[i] * aggregated_outputs[i] for i in range(self.num_early_exits)]
        )
        final_output = combined_output.sum(dim=0)

        return final_output, exit_probs
