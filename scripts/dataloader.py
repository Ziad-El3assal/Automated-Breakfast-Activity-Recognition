import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as io

class BreakfastActionsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for video_name in os.listdir(class_dir):
                    video_path = os.path.join(class_dir, video_name)
                    if os.path.isfile(video_path):
                        self.video_paths.append(video_path)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        video_frames, _, _ = io.read_video(video_path, pts_unit='sec')

        if self.transform:
            video_frames = self.transform(video_frames)

        return video_frames, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomCrop((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

batch_size = 32
train_dir = 'Breakfast Action Recognition\\train'
valid_dir = 'Breakfast Action Recognition\\valid'
# Create datasets and dataloaders
train_dataset = BreakfastActionsDataset(train_dir, transform=transform)
valid_dataset = BreakfastActionsDataset(valid_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
