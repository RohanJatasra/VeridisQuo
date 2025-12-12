import torch
import torch.nn as nn
from torchvision import models

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, path_to_weights=None, model_name='efficientnet_b4', pretrained=True):
        super(EfficientNetFeatureExtractor, self).__init__()
    
        weights = 'DEFAULT' if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)

        if path_to_weights is not None:
            self.load_weights(path_to_weights)
            
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        features = self.features(x)
        pooled_features = self.avgpool(features)
        return features, pooled_features
        
    def save_weights(self, path):
        torch.save(self.backbone.state_dict(), path)
        
    def load_weights(self, path):
        state_dict = torch.load(path)
        self.backbone.load_state_dict(state_dict)
        

