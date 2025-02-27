"""
Made by Alston Tang for AI/ML Club Countering Noise Workshop

Provides some simple transforms for primarily Image dataset, though can be modified for many more datasets.
"""

import torch
import numpy as np
from torchvision import transforms

class AddGaussianNoise():
    def __init__(self, mean=0.0, std=0.1, probability=0.5):
        self.std = std
        self.mean = mean
        self.probability = probability
        
    def __call__(self, tensor):
        if np.random.random() > self.probability:
            return tensor
        maximum, minimum = torch.max(tensor).item(), torch.min(tensor).item() # We don't want exceeding values here!
        mask = torch.rand_like(tensor).lt(self.probability).float() # Here we make a mask based on probability (random uniform distribution less than certain probability (0 to 1))
        return torch.clamp(tensor + (torch.randn_like(tensor) * self.std + self.mean)*mask, minimum, maximum) # Add standard deviation (68% of values fall within +-self.std) and shift the mean.

class RotationProbability():
    def __init__(self, degrees, probability=0.5):
        self.probability = probability
        self.transform = transforms.RandomRotation(degrees)
        
    def __call__(self, tensor):
        if np.random.random() > self.probability:
            return tensor
        return self.transform(tensor)

class GaussianBlurProbability():
    def __init__(self, blurStrength, probability=0.5):
        self.probability = probability
        self.transform = transforms.GaussianBlur(blurStrength)
        
    def __call__(self, tensor):
        if np.random.random() > self.probability:
            return tensor
        return self.transform(tensor)

def getTransform(probability=0.1):
    return transforms.Compose([
        transforms.ToTensor(), # 1, 2, 3 -> torch.tensor([1.0, 2.0, 3.0])
        transforms.RandomVerticalFlip(probability),
        transforms.RandomHorizontalFlip(probability),
        RotationProbability(45, probability),
        AddGaussianNoise(probability=probability),
        GaussianBlurProbability(3, probability)
    ])

def getIdentityTransform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

# Tomorrow: Add viewing and color-based transforms.