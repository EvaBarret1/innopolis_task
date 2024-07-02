import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(ObjectDetectionModel, self).__init__()
        self.num_classes = num_classes
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 4 * (self.num_classes + 1))
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        
        bbox_output = self.bbox_head(x)
        cls_output = self.cls_head(x)
        
        return bbox_output, cls_output
