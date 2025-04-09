from PIL import Image
import time
import pytesseract
from source.data_model import DocDataset
from sklearn.metrics import classification_report
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
    predict_ang = [] # list for predicted label 
    true_ang = []    # list for true label

    # iteration throught test data
    for idx in range(len(data)):
        image, label = data[idx] # take img from datset

        start = time.time()
        output = pytesseract.image_to_osd(image, config='--psm 0') # tesseract predict inf about img
        end = time.time()
        sum_time += end - start # count sum time

        angle = int(output.split("Rotate:")[1].split("\n")[0].strip()) # take from predicted inf omly angle

        predict_ang.append(angle)         # save predicted label
        true_ang.append(label * 90)        # save true label

        # Check if prediction is wrong
        if angle != (label * 90):
            wrong_predictions.append({      # save information about wrong prediction
                'path': data.get_path(idx),
                'predicted': angle,
                'real': label * 90
            })

    report = classification_report(true_ang, predict_ang, digits=5)
    print("\nTable for metrics for each class")
    print("-------------------------------------")
    print(report)
    print(f"Time spent: {sum_time}")
    print("-------------------------------------")

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