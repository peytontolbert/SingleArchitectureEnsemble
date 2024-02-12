import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTConfig
from SAE import ViTSAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Assuming the ViTSAE model and EarlyExitLayer classes are defined as previously provided

# Model Configuration
config = ViTConfig(image_size=224, num_hidden_layers=12, hidden_size=768, num_attention_heads=12, num_labels=10)
model = ViTSAE(config, num_classes=10, early_exit_thresholds=[0.9, 0.95, 0.99])
model.to(device)

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print(f"Exit point: {exit_point}")
        print(f"Loss: {loss.item()}")
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

print('Finished Training')