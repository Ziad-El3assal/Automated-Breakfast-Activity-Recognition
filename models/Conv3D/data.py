import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
#cv2ShowImage
def cv2ShowImage(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None,num_frames=16):
        """
        Args:
            video_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_frames = num_frames
        self.video_dir = video_dir
        self.transform = transform
        self.class_freq=dict(zip(['cereals', 'coffee', 'friedegg', 'juice', 'milk', 'pancake', 'salat', 'sandwich', 'scrambledegg', 'tea'], np.zeros(10)))
        self.class_weights = []
        self.classes=['cereals', 'coffee', 'friedegg', 'juice', 'milk', 'pancake', 'salat', 'sandwich', 'scrambledegg', 'tea']
        self.video_files = self._get_video_files()
        self.class_map =  dict(zip(['cereals', 'coffee', 'friedegg', 'juice', 'milk', 'pancake', 'salat', 'sandwich', 'scrambledegg', 'tea'], range(10)))
    def _get_video_files(self):
        video_files = []
        for root, _, files in os.walk(self.video_dir):
            for file in files:
                if file.endswith('.avi'):
                    
                    class_name = file.split('_')[1].split('.')[0]
                    self.class_freq[class_name]+=1
                    video_files.append(os.path.join(root, file))
                    
        self.class_weights = [1.0 / self.class_freq[clas] for clas in self.classes]
        print(self.class_weights)
        return video_files
    def get_class_weights(self):
        return self.class_weights
    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        video_path = self.video_files[idx]
        class_name = os.path.basename(video_path).split('_')[1].split('.')[0]
        
        label = self.class_map[class_name]
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # Sample frames
        frames = self._sample_frames(frames)
        # #show frames
        # for frame in frames:
        #     cv2ShowImage(frame)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Convert to numpy array and apply transforms
        frames = np.array(frames)
        
        #print(frames.shape)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        
        # Convert frames to torch tensor
        frames = torch.stack(frames).permute(1, 0, 2, 3)
        #print(frames.shape)
        return frames, label

    def _sample_frames(self, frames):
        num_frames = len(frames)
        if num_frames < self.num_frames:
            # Pad with the last frame
            padding = [frames[-1]] * (self.num_frames - num_frames)
            frames.extend(padding)
        elif num_frames > self.num_frames:
            # Randomly sample frames
            indices = np.linspace(0, num_frames - 1, self.num_frames).astype(int)
            #print(indices)
            frames = [frames[i] for i in indices]
        return frames

# Define transform (example)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 240)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop((112, 112)),
    transforms.ToTensor()
])
def createDataLoader(video_dir,batch_size,transform=transform,num_frames=16):
    dataset = VideoDataset(video_dir=video_dir, transform=transform,num_frames=num_frames)
    class_weights = dataset.get_class_weights()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    return data_loader,class_weights

# def main():
#     # Example usage
#     video_dir = '..\..\Breakfast Action Recognition\\train'

#     dataset = VideoDataset(video_dir=video_dir, transform=transform)
#     data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=6)
#     # Iterate through the data
#     for inputs, labels in data_loader:
#         print(inputs.shape, labels)
#         break

# if __name__ == '__main__':
#     main()