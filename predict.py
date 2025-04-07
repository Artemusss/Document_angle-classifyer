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
    image = transform(image).unsqueeze(0).to(DEVICE) # transform img
    
    with torch.no_grad(): # disable counting gradient
        start = time.time() 
        output = model(image)        # take logs
        end = time.time()
        _, predicted = torch.max(output, 1) # transform logs in class
        

        angle = predicted.item() * 90 # counting predicted angle
        confidence = torch.softmax(output, dim=1)[0][predicted].item() # counting confidence
    
    print(f'Image: {image_path}')
    print(f'Predicted angle: {angle}')
    print(f'Confidence: {confidence:.2%}')
    print(f'Time spent: {end - start}')

# Testing model on test data
def test_model(test_dir):

    model = OrientationDocCNN().to(DEVICE) # creating and connect to device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)) # load best model
    model.eval() # disable layers using for learning
    
    data = DocDataset(test_dir) # load test images
    wrong_predictions = [] # list for path to files wrong predicted
    sum_time = 0
    
    with torch.no_grad(): # disable counting gradient
        # iteration throught test data
        for idx in range(len(data)):
            image, label = data[idx]              # take img from datset
            image = image.unsqueeze(0).to(DEVICE) # transform to batch with size 1

            start = time.time()
            output = model(image) # take logs
            _, predicted = torch.max(output, 1) # transform logs in class
            end = time.time()
            sum_time += end - start # count sum time

            # Check if prediction is wrong
            if predicted.item() != label:
                wrong_predictions.append({          # save information about wrong prediction
                    'path': data.get_path(idx),
                    'predicted': predicted.item() * 90,
                    'real': label * 90
                })
    
    accuracy = 100 * (len(data) - len(wrong_predictions)) / len(data) # counting accuracy

    print(f"\nTest accuracy: {accuracy:.7f}%")
    print(f"Time spent: {sum_time}")

    print(f"Wrong predictions ({len(wrong_predictions)}):")
    for item in wrong_predictions:
        print(f"Image: {item['path']}")
        print(f"Predicted: {item['predicted']}, Real: {item['real']}\n")

# Code for testing
test_model('Train_and_test_data/test')

# Code for single prediction
#print("Write path to img:")
 
#path = input()
 
#predict(path)