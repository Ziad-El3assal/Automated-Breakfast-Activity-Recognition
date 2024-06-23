# Parameters

from model import CNN_LSTM, CNN
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append('..\..\scripts')
from dataloader import create_data_loader


sequence_length = 5
num_classes = 10  
cnn = CNN()
model = CNN_LSTM(cnn, num_classes, sequence_length)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
train_loader = create_data_loader('..\..\data\\train', batch_size=10, shuffle=False)

val_loader = create_data_loader('..\\..\\data\\valid', batch_size=10, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    model.train()
    running_loss = 0.0
    for frames, labels in train_loader:
        optimizer.zero_grad()#zero gradients to avoid accumulation of gradients
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for frames, labels in val_loader:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%')


#save model

torch.save(model.state_dict(), 'model.pth')

#save training and validation loss
import pickle
with open('train_val_loss.pkl', 'wb') as f:
    pickle.dump([running_loss/len(train_loader), val_loss/len(val_loader)], f)