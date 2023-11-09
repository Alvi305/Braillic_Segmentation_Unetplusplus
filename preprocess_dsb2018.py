import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def main():
    img_size = 256

    # Adjust the paths to match your directory structure
    img_paths = glob('X:/Braillic/Segmentation_ML/pytorch-nested-unet/inputs/BrainMRI/images/*.jpg')

    # Adjust these to where you want to save the resized images and masks
    os.makedirs('X:/Braillic/Segmentation_ML/pytorch-nested-unet/inputs/BrainMRI_%d/images' % img_size, exist_ok=True)
    os.makedirs('X:/Braillic/Segmentation_ML/pytorch-nested-unet/inputs/BrainMRI_%d/masks' % img_size, exist_ok=True)

    for path in tqdm(img_paths):
        img = cv2.imread(path)
        # Assuming masks have the same name as images, but are in the 'masks' directory and are .png files
        mask_path = path.replace('images', 'masks').replace('.jpg', '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[..., :3]
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        cv2.imwrite(os.path.join('X:/Braillic/Segmentation_ML/pytorch-nested-unet/inputs/BrainMRI_%d/images' % img_size,
                    os.path.basename(path).replace('.jpg', '.png')), img)
        cv2.imwrite(os.path.join('X:/Braillic/Segmentation_ML/pytorch-nested-unet/inputs/BrainMRI_%d/masks' % img_size,
                    os.path.basename(path).replace('.jpg', '.png')), mask)


if __name__ == '__main__':
    main()