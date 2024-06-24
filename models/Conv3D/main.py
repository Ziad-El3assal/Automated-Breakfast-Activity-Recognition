from data import createDataLoader
from model import build_model
import torch
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score






def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            print("Real Pred: ",preds)
            print("Real Labels: ",labels.data)
            # print("predictions: ",outputs)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            print("Running Loss:  ",running_loss)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    return model

def evaluate_model(model, dataloader):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    f1= f1_score(all_labels, all_preds, average='weighted')
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy,precision,f1

def main():
    num_classes = 10

    model = build_model(num_classes)


    train_loader,class_weights = createDataLoader('..\..\Breakfast Action Recognition\\train', batch_size=3, num_frames=25)

    val_loader,_ = createDataLoader('..\..\Breakfast Action Recognition\\valid', batch_size=3, num_frames=25)

    print("Data loaders created successfully")

    class_weights = torch.FloatTensor(class_weights).to('cuda:0')
    print("Class weights: ",class_weights)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=15)
    
    acc,p,f1=evaluate_model(trained_model, val_loader)
    print("Validation Accuracy: ", acc)
    print("Validation Precision: ", p)
    print("Validation F1 Score: ", f1)
    
    #save model
    torch.save(trained_model.state_dict(), 'ResNet3D.pth')
    
    print("Model saved successfully")

if __name__ == '__main__':
    main()