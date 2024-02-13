import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTConfig
from SAEv2 import ConcatenatedInputViTSAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Assuming the ViTSAE model and EarlyExitLayer classes are defined as previously provided

# Model Configuration
config = ViTConfig(
    image_size=224,
    num_hidden_layers=12,
    hidden_size=768,
    num_attention_heads=12,
    num_labels=64,
)
model = ConcatenatedInputViTSAE(
    config, num_classes=64, num_early_exits=6, num_inputs=64
)
model.to(device)

# Dataset and DataLoader
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Dataset and DataLoader - Testing
transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs, exit_point = model(images)
        # Process the outputs as needed, e.g., applying softmax to convert to probabilities
        labels = labels.float()
        print(f"output shape: {outputs.shape}")

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"Exit point: {exit_point}")
        print(f"Loss: {loss.item()}")
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluation Loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, exit_point = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Final accuracy on test set: {accuracy}%")

print("Finished Training")
