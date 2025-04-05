from PIL import Image
import torch
from source.config import DEVICE, MODEL_PATH, get_transform
from source.data_model import DocDataset, OrientationDocCNN
import time 

# Prediction for single image
def predict(image_path):

    model = OrientationDocCNN().to(DEVICE) # creating and connect to device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)) # load best model
    model.eval() # disable layers using for learning
    
    transform = get_transform()                             # get transform function
    image = Image.open(image_path).convert('L')             # open and convert img in gray
    image_tensor = transform(image).unsqueeze(0).to(DEVICE) # transform img
    
    with torch.no_grad(): # disable counting gradient
        output = model(image_tensor)        # take logs
        _, predicted = torch.max(output, 1) # transform logs in class

        angle = predicted.item() * 90 # counting predicted angle
        confidence = torch.softmax(output, dim=1)[0][predicted].item() # counting confidence
    
    print(f'Image: {image_path}')
    print(f'Predicted angle: {angle}°')
    print(f'Confidence: {confidence:.2%}')
    return angle

# Testing model on test data
def test_model(test_dir):

    model = OrientationDocCNN().to(DEVICE) # creating and connect to device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)) # load best model
    model.eval() # disable layers using for learning
    
    dataset = DocDataset(test_dir) # load test images
    wrong_predictions = [] # list for path to files wrong predicted
    sum_time = 0
    
    with torch.no_grad(): # disable counting gradient
        # iteration throught test data
        for idx in range(len(dataset)):
            image, label = dataset[idx]           # take img from datset
            image = image.unsqueeze(0).to(DEVICE) # transform to batch with size 1

            start = time.time() # take time start
            output = model(image) # take logs
            _, predicted = torch.max(output, 1) # transform logs in class
            end = time.time() # take time end
            sum_time += end - start # count sum time

            # Check if prediction is wrong
            if predicted.item() != label:
                wrong_predictions.append({          # save information about wrong prediction
                    'path': dataset.get_path(idx),
                    'predicted': predicted.item() * 90,
                    'real': label * 90
                })
    
    accuracy = 100 * (len(dataset) - len(wrong_predictions)) / len(dataset) # counting accuracy

    print(f"\nTest accuracy: {accuracy:.7f}%")
    print(f"Time spent: {sum_time}")

    print(f"Wrong predictions ({len(wrong_predictions)}):")
    for item in wrong_predictions:
        print(f"Image: {item['path']}")
        print(f"Predicted: {item['predicted']}, Real: {item['real']}\n")

# Code for testing
test_model('Test_images/test')