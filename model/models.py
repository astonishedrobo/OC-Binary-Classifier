import torch.nn as nn
from torchvision import models

class InceptionV3Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionV3Classifier, self).__init__()
        self.inception_v3 = models.inception_v3(pretrained=False)
        in_features = self.inception_v3.fc.in_features
        self.inception_v3.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.inception_v3(x)
    
class InceptionResNetV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(InceptionResNetV2Classifier, self).__init__()
        self.inceptionresnet_v2 = models.inceptionresnetv2(pretrained=True)
        in_features = self.inceptionresnet_v2.fc.in_features
        self.inceptionresnet_v2.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.inceptionresnet_v2(x)
