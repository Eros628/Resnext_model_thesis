import torch.nn as nn
from torchvision import models

class ResNeXtTeacher(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load a pretrained ResNeXt model
        try:
            # Modern torchvision API
            self.model = models.resnext50_32x4d(weights='IMAGENET1K_V1')
        except TypeError:
            # Fallback for older torchvision
            self.model = models.resnext50_32x4d(pretrained=True)
            
        # Adapt the first convolutional layer to accept 1-channel (grayscale) LFCC input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer for our binary classification task
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)