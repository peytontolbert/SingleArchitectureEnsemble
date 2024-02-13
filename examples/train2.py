import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from SAEv2 import ViTSAE
from transformers import ViTConfig
from PIL import Image

# Ensure the checkpoint directory exists
checkpoint_path = "single_architecture_ensemble_model.pth"
os.makedirs(checkpoint_path, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transformations applied on each image
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Load CIFAR-10 dataset
dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# Split dataset into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Model Configuration
config = ViTConfig()
num_classes = 11  # CIFAR-10 classes
num_early_exits = 10
depth = 11
learning_rate = 1e-2
# Initialize the model
model = ViTSAE(config, num_classes, num_early_exits, depth)
model = model.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-2)

num_epochs, step = 10, 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        Y = labels.to(device)
        X = images.to(device)
        # print(f'labels and labels.shape before we get the output {labels, labels.shape}')
        print(X.shape)
        optimizer.zero_grad()
        outputs, exit_point = model(X)
        print(
            f"labels and outputs after we getting the outputs {Y, Y.shape, outputs, outputs.shape}"
        )
        print(f"Exit point: {exit_point}")
        print(f"Output shape: {outputs.shape}")
        # print(f'labels and outputs after we unsqueeze {labels, labels.shape, outputs, outputs.shape}')
        # print(f' ########################\ labels before converting them into long {labels, labels.dtype, labels.shape}')
        # labels = labels.float()
        # print(f' ####################\ labels after converting them into long {labels, labels.dtype, labels.shape}')
        # outputs = outputs.view(Y.shape[0], -1)
        print(outputs.shape)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"Exit point: {exit_point}")
        print(f"Loss: {loss.item()}")

        if step % 10 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}"
            )

        # Save checkpoints periodically
        if step % 100 == 0:
            checkpoint = {
                "epoch": epoch,
                "batch_idx": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }
            checkpoint_filename = f"checkpoint_epoch_{epoch}_batch_{step}.pth"
            checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
            torch.save(checkpoint, checkpoint_filepath)
            print(f"Checkpoint saved to {checkpoint_filepath}")


"""# Example forward pass with dummy inputs (replace with actual data handling in practice)
input_tensor1 = torch.rand(32, 3, 224, 224)  # Batch of 32 images
input_tensor2 = torch.rand(32, 3, 224, 224)  # Another batch of 32 images

input_tensor1 = input_tensor1.to(device)
input_tensor2 = input_tensor2.to(device)


# Combine inputs for the model
inputs = [input_tensor1, input_tensor2]

# Forward pass
final_output, exit_probs = model(inputs)

print("Final Output Shape:", final_output.shape)
print("Exit Probabilities:", exit_probs)"""

# Save the final model
final_model_path = os.path.join(
    checkpoint_path, "single_architecture_ensemble_model.pth"
)
torch.save(model.state_dict(), final_model_path)
print("Training complete. Model saved to", final_model_path)
