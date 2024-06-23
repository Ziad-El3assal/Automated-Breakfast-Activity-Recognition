import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '\\scripts')

from dataloader import create_data_loader

train_path = 'Breakfast Action Recognition\\train'
train_loader = create_data_loader(train_path, batch_size=8, shuffle=True)

test_path = 'Breakfast Action Recognition\\valid'
test_loader = create_data_loader(test_path, batch_size=8, shuffle=False)

for i in train_loader:
    print(i)
    break