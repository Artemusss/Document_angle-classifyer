import os
import sys
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import threading

"""
Programm take img from folder rotate it 4 times and 4 times saving in folders with names 0, 90, 180, 270
"""
# Function for taking rotation and save images
def rotation_and_save(file_list_elem):
    filename, file_counter = file_list_elem  # take values
    file_path = os.path.join(path_inp, filename) # create file path

    try:
        img = Image.open(file_path) # open img
        img.load() # load img in memory
        
        # four iterations for diffrent angles
        for i, folder in enumerate(path_outputs, 1):
            angle = 90 * (i - 1)                         # count result angle
            rotated_img = img.rotate(angle, expand=True) # rotate img

            output_path = os.path.join(folder, f'rotated_{angle}_{file_counter}.jpg') # make output path
            rotated_img.save(output_path, "JPEG")  # save as JPEG 
            print(f"Saved: {output_path}")

            rotated_img.close() # disable rotated img from memory

        img.close() # disable img from memory
                
    except Exception as e: # error handling
        print(f"\nError with file {filename}: {e}\n")


print("Enter path to folder with images\n")
path_inp = input()

if not os.path.isdir(path_inp): # checking existing input folder
    print(f"Error: folder '{path_inp}' doesn't exist\n")
    sys.exit(1)

print("Enter 4 paths to folders 0, 90, 180, 270 (if this folders doesn't exist they will be made)\n")
path_outputs = [] # list for output folders path
for i in range(4):
    path_outputs.append(input())

# making folders if don't exist
for folder in path_outputs:
    os.makedirs(folder, exist_ok=True)

counter = 1 # counter for numeretion img
counter_lock = threading.Lock()  # count sinc

# Create list of (filename, counter) pairs before processing
with counter_lock:  # protect counter while preparing tasks
    file_list = [(f, i) for i, f in enumerate(os.listdir(path_inp), counter)] # make list of tuples like (path, number) number starts from counter
    counter += len(file_list)  # update global counter

with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:  # for every cpu usage
    executor.map(rotation_and_save, file_list)  # use map with prepared list