import os
import sys
from PIL import Image

"""
Programm take img from folder rotate it 4 times and 4 times saving in folders with names 0, 90, 180, 270
"""
print("Enter path to folder with images\n")
path_inp = input()

if not os.path.isdir(path_inp): # checking existing input folder
    print(f"Error: folder '{path_inp}' doesn't exist\n")
    sys.exit(1)

print("Enter 4 paths to folders 0, 90, 180, 270 (if this folders doesn't exist they will be made)\n")
path_outputs = [] # list for output folders path
for i in range(4):
    path_outputs.append(input())

for folder in path_outputs:
    os.makedirs(folder, exist_ok=True)

counter = 1 # counter for numeretion img

# iterate throught input folder
for filename in os.listdir(path_inp):

    file_path = os.path.join(path_inp, filename) # create file path

    try:
            img = Image.open(file_path) # open img
            img.load() # load img in memory
            
            # four iterations for diffrent angles
            for i, folder in enumerate(path_outputs, 1):

                angle = 90 * (i - 1)                         # count result angle
                rotated_img = img.rotate(angle, expand=True) # rotate img

                output_path = os.path.join(folder, f'rotated_{angle}_{counter}.png') # make output path
                rotated_img.save(output_path, "PNG")                                 # save img by output path
                print(f"Saved: {output_path}")

                rotated_img.close() # disable rotated img from memory

            img.close() # disable img from memory
                    
    except Exception as e: # error handling
        print(f"\nError with file {filename}: {e}\n")

    counter += 1 # increasing the counter

