from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor
from SAEv2 import ViTSAE
import torch
from torchvision import transforms
from PIL import Image
import os
import json
import requests
# Directory containing images
image_dir = './images'
model2 = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model2.eval()
with open('imagenet_classes.json', 'r') as f:
    idx_to_label = [json.load(f)]
# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Load and transform images
images = []
for img_file in sorted(os.listdir(image_dir)):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert('RGB')  # Ensure image is RGB
        img_t = transform(img)
        images.append(img_t)
classes = []
# Classify images with the pre-trained model (model2)
for img_t, img_file in zip(images, os.listdir(image_dir)):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(os.path.join(image_dir, img_file)).convert("RGB")
        # Add batch dimension and classify
        img_t = img_t.unsqueeze(0)  # Add batch dimension
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():  # No need to track gradients
            outputs2 = model2(**inputs)
        
        logits = outputs2.logits
        predicted_class_idx = logits.argmax(-1).item()
      
        # Optionally, print the predicted class (if id2label is available)
        predicted_label = (
            model2.config.id2label[predicted_class_idx]
            if model2.config.id2label
            else str(predicted_class_idx)
        )
        classes.append({"image": img_file, "label": predicted_label})
        print("Predicted class index:", predicted_class_idx)
        print("Predicted label:", predicted_label)
# Example usage
config = ViTConfig(image_size=224, num_hidden_layers=12, hidden_size=768, num_attention_heads=12, num_labels=20)
model = ViTSAE(config, num_classes=20, num_early_exits=20)
images_tensor = torch.stack(images)
# Example input tensor (batch size, channels, height, width)
outputs, exit_indices = model(images_tensor)

# outputs contain the predictions
# exit_indices indicates at which early exit (or final layer) the prediction was made

# Process the outputs as needed, e.g., applying softmax to convert to probabilities
probabilities = torch.softmax(outputs, dim=-1)

# Determine the predicted class for each image
_, predicted_classes = torch.max(probabilities, dim=1)

print(f"Output shape: {outputs.shape}, Exit probabilities: {exit_indices}")

# Additional insights from your custom model
for i, img_file in enumerate(os.listdir(image_dir)):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        # Directly using class index as class name for custom model predictions; adjust if you have a specific mapping
        custom_class_name = f"Custom Class {predicted_classes[i].item()}"
        exit_point = exit_indices[i]  # This might need adjustment based on your model's output structure
        
        # Print combined insights
        print(f"Image: {img_file}, Custom Model Predicted: {custom_class_name}, Exit Point: {exit_point}")