import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from source.config import DATA_DIR, BATCH_SIZE, EPOCHS, DEVICE, MODEL_PATH
from source.data_model import DocDataset, OrientationDocCNN

def train_model():
    # Loading train/valid data
    dataset = DocDataset(DATA_DIR) # set data directory
    train_size = int(0.8 * len(dataset)) # count train size
    train_data, val_data = random_split(dataset, [train_size, len(dataset) - train_size]) # random make train and valid data
    
    # Make train and valid dataloader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model initialization
    model = OrientationDocCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss() # use cross entrophy becаuse classification
    optimizer = optim.Adam(model.parameters(), lr=0.001) # use adam becаuse fast end easy to use
    
    # Learning
    num_batches = train_size // BATCH_SIZE
    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        model.train()

        # Iteration throught batch
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE) # load img and label

            optimizer.zero_grad()             # zeroing optimizer
            outputs = model(images)           # take outputs
            loss = criterion(outputs, labels) # count loss
            loss.backward()                   # count and send back grad
            optimizer.step()                  # update params CNN

            print(f"Batch number: {i}/{num_batches}")
        
        # Validation
        model.eval() # disable layers using for learning
        correct = 0
        total = 0
        with torch.no_grad(): # disable counting gradients

            # Iteration throught batch
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE) # load images and labels

                outputs = model(images)                   # take logits
                _, predicted = torch.max(outputs.data, 1) # analisys logits and give class
                
                total += labels.size(0)                       # count number of img
                correct += (predicted == labels).sum().item() # count correct predictions
        
        accuracy = 100 * correct / total # count epoch accuracy 
        print(f'Epoch {epoch+1}/{EPOCHS}, Val Acc: {accuracy:.7f}%')
        
        # Saving model with best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_PATH)
    
    print(f'Best accuracy: {best_accuracy:.2f}%')

# Code for training
train_model()