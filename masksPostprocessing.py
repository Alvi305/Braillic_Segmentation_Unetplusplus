import os
import numpy as np
import cv2
from PIL import Image

input_images = []

input_directory = r"X:\Braillic\Segmentation_ML\pytorch-nested-unet\inputs\BrainMRI_256\images"
for img in os.listdir(input_directory):
    input_images.append(os.path.join(input_directory, img))

output_base_folder = r"X:\Braillic\Segmentation_ML\pytorch-nested-unet\outputs\Post_BrainMRI_256_NestedUNet_wDS"

for file in input_images:
    slice_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    slice_img_blur = cv2.GaussianBlur(slice_img, (3, 3), 0)
    _, img_thresh = cv2.threshold(slice_img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_close = np.ones((7, 7), np.uint8)
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    contours, _ = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(slice_img)
        cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

        filename = os.path.basename(file)
        filename_no_ext = os.path.splitext(filename)[0]
        mask_filepath = os.path.join(output_base_folder, filename_no_ext + '.jpg')

        mask_img = Image.fromarray(mask)
        mask_img.save(mask_filepath)

print("Processing done ....")