import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sae.saev2 import ConcatenatedInputViTSAE
from transformers import ViTConfig

# Ensure the checkpoint directory exists
checkpoint_path = "single_architecture_ensemble_model.pth"
os.makedirs(checkpoint_path, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transformations applied on each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split dataset into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_si
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

num_classes = 32  # CIFAR-10 classes
num_early_exits = 3
num_inputs = 2  # Assuming concatenated inputs

# Initialize the model
model = ConcatenatedInputViTSAE(config, num_classes, num_early_exits, num_inputs)
model = model.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs, step = 10, 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # print(f'labels and labels.shape before we get the output {labels, labels.shape}')

        optimizer.zero_grad()
        
        outputs, exit_point = model(images)

        # print(f'labels and outputs after we getting the outputs {labels, labels.shape, outputs, outputs.shape}')

        outputs = outputs.squeeze(0)

        # print(f'labels and outputs after we unsqueeze {labels, labels.shape, outputs, outputs.shape}')

        # print(f' ########################\ labels before converting them into long {labels, labels.dtype, labels.shape}')

        labels = labels.float()

        # print(f' ####################\ labels after converting them into long {labels, labels.dtype, labels.shape}')

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print(f"Exit point: {exit_point}")
        print(f"Loss: {loss.item()}")

        if step % 10 == 0:
           print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        # Save checkpoints periodically
        if step % 100 == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            checkpoint_filename = f'checkpoint_epoch_{epoch}_batch_{step}.pth'
            checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
            torch.save(checkpoint, checkpoint_filepath)
            print(f'Checkpoint saved to {checkpoint_filepath}')
    
        


'''# Example forward pass with dummy inputs (replace with actual data handling in practice)
input_tensor1 = torch.rand(32, 3, 224, 224)  # Batch of 32 images
input_tensor2 = torch.rand(32, 3, 224, 224)  # Another batch of 32 images

input_tensor1 = input_tensor1.to(device)
input_tensor2 = input_tensor2.to(device)


# Combine inputs for the model
inputs = [input_tensor1, input_tensor2]

# Forward pass
final_output, exit_probs = model(inputs)

print("Final Output Shape:", final_output.shape)
print("Exit Probabilities:", exit_probs)'''

# Save the final model
final_model_path = os.path.join(checkpoint_path, "single_architecture_ensemble_model.pth")
torch.save(model.state_dict(), final_model_path)
print("Training complete. Model saved to", final_model_path)
