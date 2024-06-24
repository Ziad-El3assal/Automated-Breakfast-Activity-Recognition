
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class ResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3D, self).__init__()
        
        self.resnet3d = models.video.r3d_18( progress=True,weights='R3D_18_Weights.DEFAULT')
        #self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)

    def forward(self, x):
        
    
        x= self.resnet3d(x)
        return x
def build_model(num_classes):
    return ResNet3D(num_classes)


# model=build_model(10)
# print(model)