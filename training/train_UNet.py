import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


train_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs': 25,
    'batch_size': 2,
    'learning_rate': 0.001,
    'batches_per_epoch': 64,
    'lr_decay_factor': 1
}


device = train_config['device']
model = UNet(in_channels=4, out_channels=3).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
scaler = GradScaler()

checkpoint_path = "unet_segmentation_model_optimized.pth"
model.load_state_dict(torch.load(checkpoint_path))
print(f"Loaded model weights from '{checkpoint_path}'.")


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

      
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

  
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)

        
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


additional_epochs = 20
train_losses, val_losses = [], []

for epoch in range(train_config['n_epochs']):
    print(f"Epoch [{epoch + 1}/{additional_epochs}]")


    train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, scaler)
    train_losses.append(train_loss)
    print(f"Training Loss: {train_loss:.4f}")


    val_loss = validate_one_epoch(model, val_dataloader, criterion, device)
    val_losses.append(val_loss)
    print(f"Validation Loss: {val_loss:.4f}")



torch.save(model.state_dict(), "unet_segmentation_model_optimized.pth")
print("Model saved as 'unet_segmentation_model_optimized.pth'.")


import matplotlib.pyplot as plt

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()
