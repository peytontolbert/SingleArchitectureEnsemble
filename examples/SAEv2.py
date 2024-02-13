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
    def __init__(self, config, num_classes, num_early_exits, depth):
        super(ViTSAE, self).__init__()
        self.vit = ViTModel(config)
        self.num_classes = num_classes
        self.depth = depth
        # Define early exits based on the number of early exits
        self.early_exits = nn.ModuleList(
            [
                EarlyExitLayer(config.hidden_size, num_classes)
                for _ in range(depth)
            ]
        )
        # Variational parameters θ for each exit, initialized as logits
        self.exit_logits = nn.Parameter(
            torch.zeros(depth), requires_grad=True
        )

    def forward(self, x, confidence_threshold=0.5):
        # Compute exit probabilities
        exit_probs = F.softmax(self.exit_logits / confidence_threshold, dim=0)

        # Pass concatenated input through the ViT model
        hidden_states = self.vit(x).last_hidden_state
        outputs = []
        # List to hold the maximum confidence for each exit layer
        exit_confidences = []
        control_flow = x
        hidden_states = self.vit(control_flow).last_hidden_state
        for i, exit_layer in enumerate(self.depth[:self.early_exits]):
            exit_input = hidden_states[:, 0]  # Use the [CLS] token's representation
            exit_output = exit_layer(exit_input)
            outputs.append(exit_output)
            # Check the confidence of the output
            exit_confidence = F.softmax(exit_output / confidence_threshold, dim=-1)
            max_confidence = exit_confidence.max(dim=-1)[0]
            exit_confidences.append(max_confidence)
            # Check if any confidence score exceeds the threshold for early exit
            if (max_confidence > confidence_threshold).any():
                # If so, return early for the corresponding inputs
                early_exit_output = exit_output[max_confidence > confidence_threshold]
                return early_exit_output, exit_probs[i]
        # If no early exit is taken, combine the outputs from all exits
        combined_output = torch.stack(
            [exit_probs[i] * output for i, output in enumerate(outputs)], dim=0
        )
        final_output = combined_output.sum(dim=0)
        return final_output, exit_probs
