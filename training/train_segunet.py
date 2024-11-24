from Models.SegUnet import SegUnet
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import tqdm
import sys

data_dir = sys.argv[1]

class BrainScanDataset(Dataset):
    def __init__(self, file_paths, deterministic=False):
        self.file_paths = file_paths
        if deterministic:  # To always generate the same test images for consistency
            np.random.seed(1)
        np.random.shuffle(self.file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load h5 file, get image and mask
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as file:
            image = file['image'][()]
            mask = file['mask'][()]

            # Reshape: (H, W, C) -> (C, H, W)
            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))

            # Adjusting pixel values for each channel in the image so they are between 0 and 255
            for i in range(image.shape[0]):    # Iterate over channels
                min_val = np.min(image[i])     # Find the min value in the channel
                image[i] = image[i] - min_val  # Shift values to ensure min is 0
                max_val = np.max(image[i]) + 1e-4     # Find max value to scale max to 1 now.
                image[i] = image[i] / max_val

            # Convert to float and scale the whole image
            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
    
h5_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
np.random.seed(42)
np.random.shuffle(h5_files)

# Split the dataset into train and validation sets (90:10)
split_idx = int(0.9 * len(h5_files))
train_files = h5_files[:split_idx]
val_files = h5_files[split_idx:]

# Create the train and val datasets
train_dataset = BrainScanDataset(train_files)
val_dataset = BrainScanDataset(val_files, deterministic=True)

batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_input_iterator = iter(DataLoader(val_dataset, batch_size=1, shuffle=False))

# Verifying dataloaders work
for images, masks in train_dataloader:
    print("Training batch - Images shape:", images.shape, "Masks shape:", masks.shape)
    break
for images, masks in val_dataloader:
    print("Validation batch - Images shape:", images.shape, "Masks shape:", masks.shape)
    break


def diceloss(inputs, targets, smooth):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    
    return 1 - dice

def focalloss(inputs, targets, alpha=1, gamma=2, logits=False, reduce=True):
    if logits:
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    else:
        BCE_loss = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss

    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss

def compound_loss(inputs, targets, smooth, alpha=1, gamma=2, term_1=1, term_2=1, term_3=1):
    dice = diceloss(inputs, targets, smooth)
    focal = focalloss(inputs, targets, alpha, gamma)
    crossentropy = torch.nn.CrossEntropyLoss(inputs, targets)
    return term_1 * dice + term_2 * focal + term_3 * crossentropy

def dice_loss(pred, target, smooth=1.):
    num_classes = pred.shape[1]
    dice = torch.zeros(num_classes).to(pred.device)
    for c in range(num_classes):
        intersection = (pred[:, c, :, :] * target[:, c, :, :]).sum()
        dice[c] = (2. * intersection + smooth) / (pred[:, c, :, :].sum() + target[:, c, :, :].sum() + smooth)
    return 1 - dice.mean()

def compound_dice_loss(pred, target, alpha=0.2):
    return alpha * dice_loss(pred, target) + (1 - alpha) * nn.MSELoss()(pred, target)

def tversky_loss(pred, target, alpha=1.1, beta=0.7, smooth=1e-7):
    num_classes = pred.shape[1]
    tversky = torch.zeros(num_classes).to(pred.device)
    for c in range(num_classes):
        true_positives = (pred[:, c, :, :] * target[:, c, :, :]).sum()
        false_positives = (pred[:, c, :, :] * (1 - target[:, c, :, :])).sum()
        false_negatives = ((1 - pred[:, c, :, :]) * target[:, c, :, :]).sum()
        tversky[c] = (true_positives + smooth) / (true_positives + alpha * false_negatives + beta * false_positives + smooth)
    return 1 - tversky.mean()


def compound_tversky_loss(pred, target, alpha = 0.6, beta = 0.4, gamma = 0.8):
  return alpha * tversky_loss(pred, target) + beta * nn.CrossEntropyLoss()(pred, target) + gamma * dice_loss(pred, target)

def dice_loss(pred, target, smooth=1.):
    num_classes = pred.shape[1]
    dice = torch.zeros(num_classes).to(pred.device)
    for c in range(num_classes):
        intersection = (pred[:, c, :, :] * target[:, c, :, :]).sum()
        dice[c] = (2. * intersection + smooth) / (pred[:, c, :, :].sum() + target[:, c, :, :].sum() + smooth)
    return 1 - dice.mean()

def compound_dice_loss(pred, target, alpha=0.2):
    return alpha * dice_loss(pred, target) + (1 - alpha) * nn.MSELoss()(pred, target)

def tversky_loss(pred, target, alpha=1.1, beta=0.7, smooth=1e-7):
    num_classes = pred.shape[1]
    tversky = torch.zeros(num_classes).to(pred.device)
    for c in range(num_classes):
        true_positives = (pred[:, c, :, :] * target[:, c, :, :]).sum()
        false_positives = (pred[:, c, :, :] * (1 - target[:, c, :, :])).sum()
        false_negatives = ((1 - pred[:, c, :, :]) * target[:, c, :, :]).sum()
        tversky[c] = (true_positives + smooth) / (true_positives + alpha * false_negatives + beta * false_positives + smooth)
    return 1 - tversky.mean()


def compound_tversky_loss(pred, target, alpha = 0.6, beta = 0.4, gamma = 0.8):
  return alpha * tversky_loss(pred, target) + beta * nn.CrossEntropyLoss()(pred, target) + gamma * dice_loss(pred, target)

def pixel_level_accuracy(predicted_mask, ground_truth_mask):
    correct_pixels = torch.sum(predicted_mask == ground_truth_mask).item()
    total_pixels = np.prod(predicted_mask.shape)
    return correct_pixels / total_pixels

def one_hot_c_values(mask):
    class_indices = torch.argmax(mask, dim=1)  # shape [batch_size, height, width]
    
    one_hot_mask = F.one_hot(class_indices, num_classes=mask.shape[1])  # shape [batch_size, height, width, classes]
    
    one_hot_mask = one_hot_mask.permute(0, 3, 1, 2).float() # shape [batch_size, classes, height, width]
    
    return one_hot_mask
    
    

model = SegUnet(in_channels=4, n_labels=3)
#model.load_state_dict(torch.load('seg_unet_warmup_best_loss.pth'))
#print('model loaded')

criterion = compound_tversky_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.007)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model.to(device)

best_loss = float('inf')
best_accuracy = 0
# Training loop
num_epochs = 6
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, masks) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks.to(torch.float))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Free old loss from GPU
        del loss, outputs, images, masks
        torch.cuda.empty_cache()
    total_loss /= len(train_dataloader)

    total_pixel_level_accuracy = 0
    model.eval()
    for i, (images, masks) in tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Validation {epoch+1}", leave=False):
        images = images.to(device)
        predicted_masks = model(images)
        predicted_masks = one_hot_c_values(predicted_masks)
        masks = masks.to(device)
        total_pixel_level_accuracy += pixel_level_accuracy(predicted_masks, masks)
        del images, masks, predicted_masks
    total_pixel_level_accuracy /= len(val_dataloader)

    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), 'seg_unet_reweighting_best_loss.pth')

    if total_pixel_level_accuracy > best_accuracy:
        best_accuracy = total_pixel_level_accuracy
        torch.save(model.state_dict(), 'seg_unet_reweighting_best_accuracy.pth')

    print(f"Epoch {epoch+1} - Training Loss: {total_loss}")
    print(f"Epoch {epoch+1} - Validation Pixel Level Accuracy: {total_pixel_level_accuracy}")
