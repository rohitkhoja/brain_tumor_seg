from SegFormer import SegFormer
from BrainScanDataset import BrainScanDataset
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import tqdm
import torchvision.transforms as transforms
import sys

data_dir = sys.argv[1]
    
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

# ------------------------------ Accuracy Metric ------------------------------

def pixel_level_accuracy(predicted_mask, ground_truth_mask):
    correct_pixels = torch.sum(predicted_mask == ground_truth_mask).item()
    total_pixels = np.prod(predicted_mask.shape)
    return correct_pixels / total_pixels

def one_hot_c_values(mask):
    class_indices = torch.argmax(mask, dim=1)  # shape [batch_size, height, width]
    
    one_hot_mask = F.one_hot(class_indices, num_classes=mask.shape[1])  # shape [batch_size, height, width, classes]
    
    one_hot_mask = one_hot_mask.permute(0, 3, 1, 2).float() # shape [batch_size, classes, height, width]
    
    return one_hot_mask

# ------------------------------ Loss Functions ------------------------------

def dice_loss(pred, target, smooth=1.):
    num_classes = pred.shape[1]
    dice = torch.zeros(num_classes).to(pred.device)
    for c in range(num_classes):
        intersection = (pred[:, c, :, :] * target[:, c, :, :]).sum()
        dice[c] = (2. * intersection + smooth) / (pred[:, c, :, :].sum() + target[:, c, :, :].sum() + smooth)
    return 1 - dice.mean()

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
  target_indices = torch.argmax(target, dim=1)
  return alpha * tversky_loss(pred, target) + beta * nn.CrossEntropyLoss()(pred, target_indices) + gamma * dice_loss(pred, target)

# ------------------------------ Training ------------------------------
def concat_generators(*kwargs):
	for gen in kwargs:
		yield from gen


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SegFormer(
    in_channels = 4,
    widths = [64, 128, 256, 512],
    depths = [3, 4, 6, 3],
    all_num_heads = [1, 2, 4, 8],
    patch_sizes = [7, 3, 3, 3],
    overlap_sizes = [4, 2, 2, 2],
    reduction_ratios = [8, 4, 2, 1],
    mlp_expansions = [4, 4, 4, 4],
    decoder_channels = 256,
    scale_factors = [8, 4, 2, 1],
    num_classes = 3).to(device)
# model.load_state_dict(torch.load('seg_unet_reweight_best_accuracy.pth'))
# print('model loaded')

optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

criterion = compound_tversky_loss

num_epochs = 50

lambda_tv = 0.3
lambda_ce = 0.1
lambda_dice = 0.2

best_loss = float('inf')
best_accuracy = 0

print('Warmup')
for epoch in range(5):
    for i, (images, masks) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}", leave=False):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        pred_mask, bottleneck = model(images)

        loss = criterion(pred_mask, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'segformer_reweight_warmup.pth')

weight_tensor = nn.Parameter(torch.ones(3)/3, requires_grad=True)
params = list(model.parameters()) + [weight_tensor]
optimizer = torch.optim.Adam(params, lr=0.004)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, masks) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}", leave=False):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        pred_mask, _ = model(images)
        
        softmax_weights = F.softmax(weight_tensor, dim=0)
        w_1 = softmax_weights[0]
        w_2 = softmax_weights[1]
        w_3 = softmax_weights[2]

        loss = criterion(pred_mask, masks, w_1, w_2, w_3) + lambda_tv*(1-w_1) + lambda_ce*(1-w_2) + lambda_dice*(1-w_3)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    total_loss = total_loss / len(train_dataloader)
    
    total_pixel_level_accuracy = 0
    model.eval()
    for i, (images, masks) in tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Validation {epoch+1}", leave=False):
        images = images.to(device)
        predicted_masks, _ = model(images)
        predicted_masks = one_hot_c_values(predicted_masks)
        masks = masks.to(device)
        total_pixel_level_accuracy += pixel_level_accuracy(predicted_masks, masks)
        del images, masks, predicted_masks
    total_pixel_level_accuracy /= len(val_dataloader)

    print(f'Epoch {epoch+1} - Loss: {total_loss:.4f}')
    print(f'Epoch {epoch+1} - Pixel Level Accuracy: {total_pixel_level_accuracy:.4f}')
    softmax_weights = F.softmax(weight_tensor, dim=0)
    print(f'Epoch {epoch+1} - Weights: {softmax_weights}')

    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), 'segformer_reweight_tensor_best_loss.pth')
    
    if total_pixel_level_accuracy > best_accuracy:
        best_accuracy = total_pixel_level_accuracy
        torch.save(model.state_dict(), 'segformer_reweight_tensor_best_accuracy.pth')