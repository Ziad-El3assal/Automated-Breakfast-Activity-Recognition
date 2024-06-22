import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as io
import PIL

class BreakfastActionsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_paths = []
        self.labels = []
        self.samples = sorted(os.listdir(root_dir))
        
        self.labelDict=dict(zip(['cereals','coffee','friedegg','juice','milk','pancake','salat','sandwich','scrambledegg','tea'],range(10)))
        for person in self.samples:
            label=person.split('_')[1]
            label=self.labelDict[label]
            for frame in os.listdir(os.path.join(root_dir,person)):
                self.video_paths.append(os.path.join(root_dir,person,frame))
                self.labels.append(label)
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        frame_path = self.video_paths[idx]
        label = self.labels[idx]
        frame= PIL.Image.open(frame_path)

        if self.transform:
            frame = self.transform(frame)

        return frame, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomCrop((28, 28)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# tst = BreakfastActionsDataset('Breakfast Action Recognition\\train')


# def create_data_loader(root_dir, batch_size=1, shuffle=False):
#     dataset = BreakfastActionsDataset(root_dir, transform=transform)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# df= create_data_loader('..\data\\train')
# for i in df:
#     print(i)
#     break
print('DataLoader loaded')