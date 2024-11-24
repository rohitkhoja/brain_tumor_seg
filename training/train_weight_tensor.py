from Models.SegUnet import SegUnet
from Models.subnet import Subnet
from Dataset.BrainScanDataset import BrainScanDataset
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


def compound_tversky_loss(pred, target, w1 = 0.6, w2 = 0.4, w3 = 0.8, tv_alpha=1.1, tv_beta=0.7):
    target_indices = torch.argmax(target, dim=1)
    return w1 * tversky_loss(pred, target, tv_alpha, tv_beta) + w2 * nn.CrossEntropyLoss()(pred, target_indices) + w3 * dice_loss(pred, target)

# ------------------------------ Training ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SegUnet(3, 3, return_bottleneck=True).to(device)
model.load_state_dict(torch.load('seg_unet_pretrain_compound_loss.pth'))
print('model loaded')

weight_tensor = nn.Parameter(torch.ones(3, device=device) / 3, requires_grad=True)

# tv_weight_tensor = nn.Parameter(torch.ones(2)/2, requires_grad=True)
# tv_weight_tensor = tv_weight_tensor.to(device)

params = list(model.parameters()) + [weight_tensor]# + [tv_weight_tensor]

optimizer = torch.optim.Adam(params, lr=0.004)

criterion = compound_tversky_loss

num_epochs = 50

lambda_tv = 0.33
lambda_ce = 0.15
lambda_dice = 0.3


for epoch in range(num_epochs):
    
    total_loss = 0
    total_train_accuracy = 0
    for i, (images, masks) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}", leave=False):
        model.train()
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        pred_mask, bottleneck = model(images)
        
        softmax_weights = F.softmax(weight_tensor, dim=0)
        w_1 = softmax_weights[0]
        w_2 = softmax_weights[1]
        w_3 = softmax_weights[2]

        # softmax_alpha_beta = F.softmax(tv_weight_tensor, dim=0)
        # alpha = softmax_alpha_beta[0]
        # beta = softmax_alpha_beta[1]

        regularization = lambda_tv*(1-w_1) + lambda_ce*(1-w_2) + lambda_dice*(1-w_3) + 1/3 * ((w_1-1/3)**2 + (w_2-1/3)**2 + (w_3-1/3)**2)

        loss = criterion(pred_mask, masks, w_1, w_2, w_3) + regularization
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        one_hot_pred = one_hot_c_values(pred_mask)

        total_train_accuracy += pixel_level_accuracy(one_hot_pred, masks)
        total_loss += loss.item()
        del loss, pred_mask, bottleneck, images, masks, one_hot_pred

    total_loss /= len(train_dataloader)
    total_train_accuracy /= len(train_dataloader)
    
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
    print(f'Epoch {epoch+1} - Train Pixel Level Accuracy: {total_train_accuracy:.4f}')
    print(f'Epoch {epoch+1} - Val Pixel Level Accuracy: {total_pixel_level_accuracy:.4f}')
    softmax_weights = F.softmax(weight_tensor, dim=0)
    print(f'Epoch {epoch+1} - Weights: {softmax_weights}')


