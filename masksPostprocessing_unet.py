import os
import io
from PIL import Image
from rembg import remove

input_images = []
input_directory = r"X:\Braillic\Segmentation_ML\pytorch-nested-unet\outputs\BrainMRI_256_NestedUNet_wDS\0"
for img in os.listdir(input_directory):
    input_images.append(os.path.join(input_directory, img))

output_base_folder = r"X:\Braillic\Segmentation_ML\pytorch-nested-unet\outputs\Post_BrainMRI_256_NestedUNet_wDS"

for file in input_images:
    with open(file, 'rb') as img_file:
        img = img_file.read()
        output = remove(img)
        
        output_img = Image.open(io.BytesIO(output))

        # Convert to grayscale
        output_img_gray = output_img.convert('L')
        
        filename = os.path.basename(file)
        filename_no_ext = os.path.splitext(filename)[0]
        mask_filepath = os.path.join(output_base_folder, filename_no_ext + '.jpg')

        output_img_gray.save(mask_filepath)
    
print("Processing Done ...")