import json
import numpy as np
from PIL import Image
import os
import cv2

# Directory containing the JSON files
directory = "X:\Braillic\Segmentation_ML\DataSet\Validation\masks"

# Iterate over the files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .json file
    if filename.endswith(".json"):
        # Construct the full filepath
        filepath = os.path.join(directory, filename)
        
        # Open the JSON file
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Create an empty array for the mask
        mask = np.zeros((256, 256), dtype=np.uint8) # Adjust the size as needed

        # Iterate over the shapes
        for shape in data['shapes']:
            # Check if the label is "foreground"
            if shape['label'] == 'foreground':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], color=1)

        # Save the mask as a .png image
        mask_image = Image.fromarray(mask*255) # multiply by 255 to get a black and white image
        mask_image.save(filepath.replace('.json', '.png'))