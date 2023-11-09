import os
import cv2
from PIL import Image

# List for storing input images and masks
input_images = []
masks = []

# Input Folder
input_directory = r"X:\Braillic\Segmentation_ML\pytorch-nested-unet\inputs\BrainMRI_256\images"
for img in os.listdir(input_directory):
    input_images.append(img)

# Folder for the masks
masks_dir = r"X:\Braillic\Segmentation_ML\pytorch-nested-unet\outputs\Post_BrainMRI_256_NestedUNet_wDS"
for msks in os.listdir(masks_dir):
    masks.append(msks)

# Output Folder where images will be saved
output_base_folder = r"X:\Braillic\Segmentation_ML\pytorch-nested-unet\foregroundimages"

for file in masks:
    # Ensure that the corresponding image file exists
    corresponding_image_file = file
    if corresponding_image_file not in input_images:
        print(f"No corresponding image for mask {file}")
        continue

    # Load mask
    mask_img = cv2.imread(os.path.join(masks_dir, file), cv2.IMREAD_GRAYSCALE)

    # Load corresponding image
    image = cv2.imread(os.path.join(input_directory, corresponding_image_file), cv2.IMREAD_GRAYSCALE)
    img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Subtract mask from the image to get the foreground
    background = cv2.subtract(image, mask_img)
    foreground = cv2.subtract(image, background)

    # Save the result
    filename = os.path.basename(file)
    filename_no_ext = os.path.splitext(filename)[0]
    
    foreground = Image.new("RGBA", mask_img.shape[::-1], (0, 0, 0, 0))  # Create new blank (transparent) image
    image_pil = Image.fromarray(img_8bit).convert("RGBA") 
    mask_img_pil = Image.fromarray(mask_img)  # Convert the mask to a PIL Image object
    foreground.paste(image_pil, mask=mask_img_pil)  # Paste in the segmented part using the mask

    # Save with filename
    foreground.save(os.path.join(output_base_folder, f"{filename_no_ext}.png"))

print("Processing done ....")