import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import StochasticDepth
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR
import tqdm
from einops import rearrange
from typing import List
import numpy as np

class LayerNorm2d(nn.LayerNorm):
  def forward(self, x):
    x = rearrange(x, "b c h w -> b h w c")
    x = super().forward(x)
    x = rearrange(x, "b h w c -> b c h w")
    return x

class OverlapPatchMerging(nn.Module):
  def __init__(self, in_channels, out_channels, patch_size, overlap_size):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = patch_size, stride = overlap_size, padding = patch_size // 2, bias = False)
    self.layer_norm = LayerNorm2d(out_channels)

  def forward(self, x):
    x = self.conv(x)
    x = self.layer_norm(x)
    return x

class EfficientMultiHeadAttention(nn.Module):
  def __init__(self, channels, reduction_ratio, num_heads = 8):
    super().__init__()
    self.reducer = nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size = reduction_ratio, stride = reduction_ratio),
        LayerNorm2d(channels)
    )
    self.attention = nn.MultiheadAttention(channels, num_heads = num_heads, batch_first = True)

  def forward(self, x):
    _, _, h, w = x.shape
    reduced_x = self.reducer(x)
    reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
    x = rearrange(x, "b c h w -> b (h w) c")
    out = self.attention(x, reduced_x, reduced_x)[0]
    out = rearrange(out, "b (h w) c -> b c h w", h = h, w = w)
    return out

class MixMLP(nn.Module):
  def __init__(self, channels, expansion = 4):
    super().__init__()
    self.conv_1 = nn.Conv2d(channels, channels, kernel_size = 1)
    self.conv_2 = nn.Conv2d(channels, channels * expansion, kernel_size = 3, groups = channels, padding = 1)
    self.gelu = nn.GELU()
    self.conv_3 = nn.Conv2d(channels * expansion, channels, kernel_size = 1)

  def forward(self, x):
    x = self.conv_1(x)
    x = self.conv_2(x)
    x = self.gelu(x)
    x = self.conv_3(x)
    return x

class ResidualAdd(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x, **kwargs):
    out = self.fn(x, **kwargs)
    x = x + out
    return x

def chunks(data, sizes):
  curr = 0
  for size in sizes:
    chunk = data[curr: curr + size]
    curr += size
    yield chunk

class SegFormerEncoderBlock(nn.Module):
  def __init__(self, channels, reduction_ratio = 1, num_heads = 8, mlp_expansion = 4, drop_path_prob = 0.0):
    super().__init__()
    self.layer_1 = ResidualAdd(nn.Sequential(
        LayerNorm2d(channels),
        EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
    ))
    self.layer_2 = ResidualAdd(nn.Sequential(
        LayerNorm2d(channels),
        MixMLP(channels, expansion = mlp_expansion),
        StochasticDepth(p = drop_path_prob, mode = "batch")
    ))
  def forward(self, x):
    x = self.layer_1(x)
    x = self.layer_2(x)
    return x

class SegFormerEncoderStage(nn.Module):
  def __init__(self, in_channels, out_channels, patch_size, overlap_size, drop_probs, depth = 2, reduction_ratio = 1, num_heads = 8, mlp_expansion = 4):
    super().__init__()
    self.overlap_patch_merge = OverlapPatchMerging(in_channels, out_channels, patch_size, overlap_size)
    self.blocks = nn.Sequential(
        *[
            SegFormerEncoderBlock(out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]) for i in range(depth)
        ]
    )
    self.norm = LayerNorm2d(out_channels)

  def forward(self, x):
    x = self.overlap_patch_merge(x)
    x = self.blocks(x)
    x = self.norm(x)
    return x

class SegFormerEncoder(nn.Module):
  def __init__(self, in_channels, widths, depths, all_num_heads, patch_sizes, overlap_sizes, reduction_ratios, mlp_expansions, drop_prob = .0):
    super().__init__()
    drop_probs = [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
    self.stages = nn.ModuleList(
        [
            SegFormerEncoderStage(*args)
            for args in zip([in_channels, *widths], widths, patch_sizes, overlap_sizes, chunks(drop_probs, sizes = depths), depths, reduction_ratios, all_num_heads, mlp_expansions)
        ]
    )
  def forward(self, x):
    features = []
    for stage in self.stages:
      x = stage(x)
      features.append(x)
    return features

class SegFormerDecoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, scale_factor = 2):
    super().__init__()
    self.upsample = nn.UpsamplingBilinear2d(scale_factor = scale_factor)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

  def forward(self, x):
    x = self.upsample(x)
    x = self.conv(x)
    return x

class SegFormerDecoder(nn.Module):
  def __init__(self, out_channels, widths, scale_factors):
    super().__init__()
    self.stages = nn.ModuleList(
        [
            SegFormerDecoderBlock(in_channels, out_channels, scale_factor) for in_channels, scale_factor in zip(widths, scale_factors)
        ]
    )

  def forward(self, features):
    new_features = []
    for feature, stage in zip(features, self.stages):
      x = stage(feature)
      new_features.append(x)
    return new_features

class SegFormerSegmentationHead(nn.Module):
  def __init__(self, channels, num_classes, num_features = 4):
    super().__init__()
    self.fuse = nn.Sequential(
        nn.Conv2d(channels * num_features, channels, kernel_size = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(channels)
    )
    self.predict = nn.Conv2d(channels, num_classes, kernel_size = 1)

  def forward(self, features): 
    x = torch.cat(features, dim = 1)
    x = self.fuse(x)
    x = self.predict(x)
    return x

class SegFormer(nn.Module):
  def __init__(self, in_channels, widths, depths, all_num_heads, patch_sizes, overlap_sizes, reduction_ratios, mlp_expansions, decoder_channels, scale_factors, num_classes, drop_prob = 0.0):
    super().__init__()
    self.encoder = SegFormerEncoder(in_channels, widths, depths, all_num_heads, patch_sizes, overlap_sizes, reduction_ratios, mlp_expansions, drop_prob)
    self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
    self.head = SegFormerSegmentationHead(decoder_channels, num_classes, num_features = len(widths))

  def forward(self, x):
    features = self.encoder(x)
    features = self.decoder(features[::-1])
    segmentation = self.head(features)
    return F.softmax(segmentation, dim = 1), features