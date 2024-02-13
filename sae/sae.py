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
    def __init__(self, config, num_classes, early_exit_thresholds):
        super(ViTSAE, self).__init__()
        self.vit = ViTModel(config)
        self.num_classes = num_classes
        self.early_exit_thresholds = early_exit_thresholds

        # Define early exits based on the number of thresholds provided
        self.early_exits = nn.ModuleList(
            [
                EarlyExitLayer(config.hidden_size, num_classes)
                for _ in early_exit_thresholds
            ]
        )

    def forward(self, x):
        outputs = []
        control_flow = x
        hidden_states = self.vit(control_flow).last_hidden_state

        for i, (exit_layer, threshold) in enumerate(
            zip(self.early_exits, self.early_exit_thresholds)
        ):
            # For simplicity, we're using the [CLS] token's representation as input to each early exit
            exit_input = hidden_states[:, 0]
            exit_output = exit_layer(exit_input)
            outputs.append(exit_output)

            # Check if we can exit early
            if torch.max(F.softmax(exit_output, dim=-1)) > threshold:
                return exit_output, i  # Early exit

        # If no early exit, return the final layer's output
        final_output = outputs[
            -1
        ]  # In practice, you might have a separate final classifier
        return final_output, len(self.early_exits)
