import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as io
from torch.nn.functional import one_hot
from torch import tensor
from PIL import Image

class BreakfastActionsDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=10):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.video_paths = []
        self.labels = []
        self.samples = sorted(os.listdir(root_dir))
        self.labelDict = dict(zip(['cereals', 'coffee', 'friedegg', 'juice', 'milk', 'pancake', 'salat', 'sandwich', 'scrambledegg', 'tea'], range(10)))
        
        for person in self.samples:
            label = person.split('_')[1]
            label = self.labelDict[label]
            frames = sorted(os.listdir(os.path.join(root_dir, person)))
            for i in range(0, len(frames) - self.sequence_length + 1):
                frame_paths = frames[i:i + self.sequence_length]
                self.video_paths.append([os.path.join(root_dir, person, frame) for frame in frame_paths])
                self.labels.append(label)
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        frame_paths = self.video_paths[idx]
        frames = []
        for frame_path in frame_paths:
            image = Image.open(frame_path)
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        frames = torch.stack(frames)
        
        label = self.labels[idx]
        return frames, label


    
        

# Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomCrop((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# tst = BreakfastActionsDataset('Breakfast Action Recognition\\train')


def create_data_loader(root_dir, batch_size=1, shuffle=False, transform=transform,sequence_length=10):
    dataset = BreakfastActionsDataset(root_dir, transform=transform,sequence_length=sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# df= create_data_loader('..\..\data\\train')
# for i in df:
#     print(i[1])
#     break
print('DataLoader loaded')