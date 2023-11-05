import torch
import torch.nn as nn
import torch.optim as optim

from unet import UNet
from dataloader import data_loader
from loss import CustomCrossEntropyLoss
import time
from sklearn.metrics import accuracy_score






in_channels = 3
num_classes = 1

model = UNet(in_channels, num_classes)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)



checkpoint_interval = 1
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, masks in data_loader:
        images = images.to(device)
        optimizer.zero_grad()

        output = model(images)
        
        
        
        loss = criterion(output.squeeze(), masks.squeeze(1))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}] Batch Loss: {loss.item()}")
        

    print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss / len(data_loader)}')

    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = f"checkpoint_epoch{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

model.eval()


total_inference_time = 0.0

with torch.no_grad():
    for images, masks in data_loader:  # Use the test data loader
        images, labels = images.to(device), masks.to(device)
        start_time = time.time()
        outputs = model(images)
        end_time = time.time()
        inference_time = end_time - start_time
        total_inference_time += inference_time
        

print(f"Total inference time: {total_inference_time} seconds")

print("fininshed training")

model = model.to(torch.float16)

model_save_path = "/home/rteam1/bazazpour/Semantic-Segmentation-UNet/quantized_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"quantized model saved to {model_save_path}")