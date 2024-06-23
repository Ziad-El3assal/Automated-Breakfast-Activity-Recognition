import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (frame_height // 8) * (frame_width // 8), 512)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return x

# Assuming frame dimensions
frame_height = 64
frame_width = 64
channels = 3
class CNN_LSTM(nn.Module):
    def __init__(self, cnn, num_classes, sequence_length):
        super(CNN_LSTM, self).__init__()
        self.cnn = cnn
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        cnn_out = []
        for i in range(seq_length):
            cnn_out.append(self.cnn(x[:, i, :, :, :]))
        cnn_out = torch.stack(cnn_out, dim=1)  # Shape: (batch_size, seq_length, 512)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out[:, -1, :]  # Take the last output of LSTM
        x = torch.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
