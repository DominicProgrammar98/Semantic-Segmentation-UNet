import torch
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet
from dataloader import data_transform

in_channels = 3
# Load your trained model
model = UNet(in_channels, num_classes=1)
model.load_state_dict(torch.load('/home/rteam1/bazazpour/Semantic-Segmentation-UNet/model.pth'))
model.eval()

# Perform inference on a sample image
sample_image = Image.open('/home/rteam1/bazazpour/Semantic-Segmentation-UNet/inference/crack-o-1.jpg')
sample_image_tensor = data_transform(sample_image).unsqueeze(0)  # Add batch dimension if needed
with torch.no_grad():
    predicted_masks = model(sample_image_tensor)

# Convert predicted_masks to class labels (e.g., using argmax)
predicted_class_labels = predicted_masks.argmax(dim=1).squeeze().cpu().numpy()

# Visualize the original image
plt.figure(figsize=(8, 8))
plt.imshow(sample_image)
plt.title("Original Image")

# Visualize the segmentation mask
plt.figure(figsize=(8, 8))
plt.imshow(predicted_class_labels, cmap='jet')  # Adjust the colormap as needed
plt.title("Segmentation Mask")

plt.show()
# Save the original image
sample_image.save('sample_image.jpg')

# Save the segmentation mask
plt.imsave('segmentation_mask.jpg', predicted_class_labels, cmap='jet')
