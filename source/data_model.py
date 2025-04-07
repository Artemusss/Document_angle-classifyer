import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from source.config import get_transform
import time

class DocDataset(Dataset):
    """
    Document dataset class

    Save documents by path and label

    When accecing the element returns transformed img and label
    """

    # Initialization
    def __init__(self, root_dir, transform=get_transform()):
        
        self.transform = transform           # save transform function
        self.classes = {'0': 0, '90': 1,
                         '180': 2, '270': 3} # dict for mapping angle to number
        self.samples = []                    # list for saving paths and labels
        
        # Iterations throught dict elements
        for i, (class_name, class_idx) in enumerate(self.classes.items()):

            class_dir = os.path.join(root_dir, class_name) # make path to folder with img

            print('\033[92mNumber of loading folder:\033[0m', i)
            time.sleep(2)

            # Iterations throught images in folder
            for j, img_name in enumerate(os.listdir(class_dir)):
                print('Number of load file:', j)

                img_path = os.path.join(class_dir, img_name) # make img path
                self.samples.append((img_path, class_idx))   # save path and dict label
    
    # Function returns number of img
    def __len__(self):
        return len(self.samples)
    
    # Function for getting img and label 
    def __getitem__(self, idx):

        img_path, label = self.samples[idx]       # get image path and label
        image = Image.open(img_path).convert('L') # open and convert image to gray
        image = self.transform(image)             # apply transform
        return image, label
    
    # Function for getting path of img
    def get_path(self, idx):
        return self.samples[idx][0]

class OrientationDocCNN(nn.Module):
    """
    CNN for classify document orientation

    Include: 3 convolution blocks, normalization, ReLU, MaxPooling
    """

    # Initialization
    def __init__(self, num_classes=4):
        super().__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), # save size
            nn.BatchNorm2d(8), # normalization activations 
            nn.ReLU(),          # activation
            nn.MaxPool2d(2),    # reduces spatial dimensions by 2
            
            # Same as first
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Same as first
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # conversion to 1D vector
            nn.Linear(32 * 16 * 16, 128), # fully connected layer
            nn.ReLU(),                    # activasion
            nn.Dropout(0.5),              # regularization to prevent overfitting
            nn.Linear(128, num_classes)   # exit layer
        )

    # Forward pass of network
    def forward(self, x):
        x = self.features(x) # feature extraction

        return self.classifier(x) # classification