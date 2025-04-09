import torch
from torchvision import transforms

"""
This is the parameter setting module
"""

# Data parameters
DATA_DIR = "Train_and_test_data" # directory with folders with images for traning/validation
BATCH_SIZE = 128          # batch size for lerning/validation
EPOCHS = 10               # number of training epochs

# Device ans save parameters
DEVICE = torch.device('cpu')              # selected device
MODEL_PATH = 'best_orientation_model.pth' # name of best model

# Transform function
def get_transform(target_size=128):

    # Make img squre size
    return transforms.Compose([
        transforms.Resize(target_size),     # reducing img size with saving proportions
        transforms.CenterCrop(target_size), # cutting img
        transforms.ToTensor(),              # transforms to tensor
        transforms.Normalize(mean=[0.485],
                              std=[0.229])  # normalaze with defoult parameters for grey
    ])