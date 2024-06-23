import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pytorchvideo
print(torch.__version__)
print(torchvision.__version__)
print(pytorchvideo.__version__)
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import UniformClipSampler
from pytorchvideo.transforms import ApplyTransformToKey
from pytorchvideo.data.encoded_video import EncodedVideo
# from pytorchvideo.data.labeled_video_paths import labeled_video_paths_from_csv

# Parameters
root_dir = '..\..\Breakfast Action Recognition'  # Directory with video frame folders
sequence_length = 10
frame_height = 64
frame_width = 64
batch_size = 8

# Define the transform
transform = Compose([
    ApplyTransformToKey(
        key="video",
        transform=Compose([
            Resize((frame_height, frame_width)),
            ToTensor()
        ])
    )
])

def get_labeled_video_paths(root_dir):
    label_dict = dict(zip(['cereals', 'coffee', 'friedegg', 'juice', 'milk', 'pancake', 'salat', 'sandwich', 'scrambledegg', 'tea'], range(10)))
    video_paths = []
    print(root_dir)
    for person in sorted(os.listdir(root_dir)):
        for video in os.listdir(os.path.join(root_dir, person)):
            label = video.split('_')[1].split('.')[0]
            label = label_dict[label]
            video_path = os.path.join(root_dir, person, video)
            video_paths.append((video_path, label))
    return video_paths

traindata = get_labeled_video_paths(os.path.join(root_dir, 'train'))

valdata = get_labeled_video_paths(os.path.join(root_dir, 'valid'))



clip_sampler = UniformClipSampler(clip_duration=2.0)



train_dataset = LabeledVideoDataset(
    labeled_video_paths=traindata,
    clip_sampler=clip_sampler,
    video_path_prefix=root_dir,
    transform=transform,
    decode_audio=False
)

val_dataset = LabeledVideoDataset(
    labeled_video_paths=valdata,
    clip_sampler=clip_sampler,
    video_path_prefix=root_dir,
    transform=transform,
    decode_audio=False
)

# trainDataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# valDataloader=DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def getdataset(type,batch_size,transform=transform):
    if type=='train':
        train_dataset = LabeledVideoDataset(
                         labeled_video_paths=traindata,
                         clip_sampler=clip_sampler,
                         video_path_prefix=root_dir,
                         transform=transform,
                         decode_audio=False
                    )

        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_dataset = LabeledVideoDataset(
                        labeled_video_paths=valdata,
                        clip_sampler=clip_sampler,
                        video_path_prefix=root_dir,
                        transform=transform,
                        decode_audio=False
                    )
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=True)