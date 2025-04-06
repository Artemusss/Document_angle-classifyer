import os
import time
from PIL import Image
import pytesseract
from source.data_model import DocDataset

# Function for predition for single document
def predict(image_path):
    image = Image.open(image_path) # openimg by path
    
    start = time.time()
    osd_data = pytesseract.image_to_osd(image) # tesseract predict information about img
    end = time.time()
    
    angle = int(osd_data.split("Rotate:")[1].split("\n")[0].strip()) # take from predict info only rotation angle

    print(f'Image: {image_path}')
    print(f'Predicted angle: {angle}')
    print(f'Time spent: {start - end}')


# Fuction for testing tesseract on test img
def test_model(test_dir):
    wrong_predictions = []  # 
    sum_time = 0
    data = DocDataset(test_dir)

    for idx in range(len(data)):
        image, label = data[idx]              # take img from datset

        start = time.time() # take time start
        pred_inf = pytesseract.image_to_osd(image) # tesseract predict information about img
        end = time.time() # take time end
        sum_time += end - start # count sum time
        pred_ang = int(pred_inf.split("Rotate:")[1].split("\n")[0].strip()) # take from predict info only rotation angle

        # Check if prediction is wrong
        if pred_ang != (label * 90):
            wrong_predictions.append({          # save information about wrong prediction
                'path': data.get_path(idx),
                'predicted': pred_ang,
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
test_model('Test_images/test')

# Code for single prediction
#print("Write path to img:")
 
#path = input()
 
#predict(path)
