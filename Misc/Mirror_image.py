import cv2
import os

# Input and output folders
input_folder = r'/home/rayuga/Documents/Project_IEM_Sort/right'
output_folder = r'/home/rayuga/Documents/Project_IEM_Sort/left'

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):

    input_path = os.path.join(input_folder, filename)

    original_image = cv2.imread(input_path)

    flipped_image = cv2.flip(original_image, 1)  

    output_path = os.path.join(output_folder, f"flip_{filename}")

    cv2.imwrite(output_path, flipped_image)

