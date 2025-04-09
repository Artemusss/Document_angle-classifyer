import os
import sys
from PIL import Image
import time
import pytesseract
from source.data_model import DocDataset
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Tesseract prediction for single image
def predict(image_path):

    image = Image.open(image_path).convert('RGB') # open and convert img in RGB
    

    start = time.time() 
    output = pytesseract.image_to_osd(image, config='--psm 0') # tesseract predict inf about img
    end = time.time()
    angle = int(output.split("Rotate:")[1].split("\n")[0].strip()) # take from predicted inf omly angle
    
    print(f'Image: {image_path}')
    print(f'Predicted angle: {angle}')
    print(f'Time spent: {end - start}')

# Testing tesseract on test data
def test_model(test_dir):
    
    data = DocDataset(test_dir, lambda img: img) # load test images
    wrong_predictions = [] # list for path to files wrong predicted
    sum_time = 0
    

    # iteration throught test data
    for idx in range(len(data)):
        image, label = data[idx] # take img from datset

        start = time.time()
        output = pytesseract.image_to_osd(image, config='--psm 0') # tesseract predict inf about img
        end = time.time()
        sum_time += end - start # count sum time

        angle = int(output.split("Rotate:")[1].split("\n")[0].strip()) # take from predicted inf omly angle

        # Check if prediction is wrong
        if angle != (label * 90):
            wrong_predictions.append({      # save information about wrong prediction
                'path': data.get_path(idx),
                'predicted': angle,
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
test_model('Train_and_test_data/compare_test')

# Code for single prediction
#print("Write path to img:")
 
#path = input()
 
#predict(path)