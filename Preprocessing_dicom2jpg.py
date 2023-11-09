import os
import numpy as np
import pydicom
from PIL import Image, ImageFilter

dicom_dir = r"X:\Braillic\Segmentation_ML\Dicom_Files"
training_dir = r"X:\Braillic\Segmentation_ML\Training_Dataset"

dicom_files = [os.path.join(dicom_dir, file) for file in os.listdir(dicom_dir)]

for dir in dicom_files:
    path_parts = dir.split(os.sep)
    output_dir_part1 = path_parts[-1]
    for filename in os.listdir(dir):
        dicom_file_path = os.path.join(dir, filename)
        dataset = pydicom.dcmread(dicom_file_path)

        # Normalize pixel data to have values between 0 and 255
        image_2d = dataset.pixel_array.astype(float)
        rescaled_image = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

        # Convert to uint8 and create a PIL image
        final_image = Image.fromarray(rescaled_image.astype(np.uint8))

        # Apply Gaussian blur to the image
        blurred_image = final_image.filter(ImageFilter.GaussianBlur(radius=1))  # Adjust the radius as needed

        # Resize the image
        resized_image = blurred_image.resize((256, 256))

        # Create output filename
        output_file = output_dir_part1 + "_" +filename
        output_filename = output_file.replace('.dcm', '.jpg')
        output_filepath = os.path.join(training_dir, output_filename)

        # Save the image
        resized_image.save(output_filepath)