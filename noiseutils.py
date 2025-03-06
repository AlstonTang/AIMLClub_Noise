"""
Made by Alston Tang for AI/ML Club Countering Noise Workshop

Provides some simple transforms for primarily Image dataset, though can be modified for many more datasets.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
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

def getTransform(probability=0.1, isColor=False):
    if not isColor:
        return transforms.Compose([
            transforms.ToTensor(), # 1, 2, 3 -> torch.tensor([1.0, 2.0, 3.0])
            transforms.RandomVerticalFlip(probability),
            transforms.RandomHorizontalFlip(probability),
            RotationProbability(45, probability),
            AddGaussianNoise(probability=probability), # salt and pepper
            GaussianBlurProbability(3, probability)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(), # 1, 2, 3 -> torch.tensor([1.0, 2.0, 3.0])
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.1), # Color-specific
            transforms.RandomGrayscale(probability), # Color-specific
            transforms.RandomVerticalFlip(probability),
            transforms.RandomHorizontalFlip(probability),
            RotationProbability(45, probability),
            AddGaussianNoise(probability=probability), # salt and pepper
            #GaussianBlurProbability(3, probability)
        ])
    
def getFMNISTPracticeTransformValidation(probability=0.1):
    return transforms.Compose([
        transforms.ToTensor(), # 1, 2, 3 -> torch.tensor([1.0, 2.0, 3.0])
        transforms.RandomVerticalFlip(probability),
        transforms.RandomHorizontalFlip(probability),
        transforms.RandomPerspective(distortion_scale=0.5, p=probability),
        transforms.RandomInvert(probability),
        RotationProbability(45, probability),
        AddGaussianNoise(0, 0.25, probability), # salt and pepper
        GaussianBlurProbability(5, probability)
    ])

def getIdentityTransform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

# Today: Add viewing
def previewCIFAR(dataset): #Should work on color
    try:
        while True:
            index = np.random.randint(0, len(dataset.data))
            plt.imshow(np.transpose(dataset[index][0].numpy(), (1,2,0)))
            plt.title(dataset.targets[index]) # torch.tensor([1]) -> 1
            plt.show()
    except KeyboardInterrupt:
        plt.close()

def previewMNIST(dataset): #Should work on grayscale
    try:
        while True:
            index = np.random.randint(0, len(dataset.data))
            plt.imshow(dataset.data[index].numpy(), cmap='gray')
            plt.title(dataset.targets[index].item()) # torch.tensor([1]) -> 1
            plt.show()
    except KeyboardInterrupt:
        plt.close()