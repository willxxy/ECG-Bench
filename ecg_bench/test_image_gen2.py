import numpy as np
import os
import random
from PIL import Image
import cv2
from imgaug import augmenters as iaa
import cv2
import glob
import heapq
from skimage import util
import time


# Original code for loading and processing ECG data
path_to_npy = glob.glob('./data/mimic/preprocessed_1250_250/*.npy')[0]

test_file = np.load(path_to_npy, allow_pickle = True).item()
print(test_file.keys())
ecg = test_file['ecg']
print(ecg.shape)

lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

from utils.viz_utils import VizUtil
VizUtil.plot_2d_ecg(ecg, '', './pngs/test_img', 250)

image = VizUtil.get_plot_as_image(ecg, 250)
print(image.shape)

# Save the original image
img = Image.fromarray(image)
original_image_path = './pngs/test_from_array.png'
img.save(original_image_path)

def augment_ecg_image(image: np.ndarray) -> np.ndarray:
    # - Multiply: randomly scales pixel intensities (mimicking lighting changes).
    # - Affine: rotates the image by a small random angle.
    # - GaussianBlur: applies a blur effect with random sigma.
    # - AddToHueAndSaturation: randomly adjusts the hue of the image
    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),    # 50% chance to change brightness
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-5, 5))),    # 50% chance to rotate by -5° to 5°
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 1.5))),  # 50% chance to apply Gaussian blur
        iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-30, 30), per_channel=True))  # 50% chance to adjust hue
    ])
    augmented_image = seq.augment_image(image)
    return augmented_image


# Example usage
augmented_image = augment_ecg_image(image)
print(f"Augmented image shape: {augmented_image.shape}")

# Save the augmented image
aug_img = Image.fromarray(augmented_image)
augmented_image_path = './pngs/augmented_ecg.png'
aug_img.save(augmented_image_path)

# Generate multiple augmentations for demonstration
for i in range(5):
    aug_img = augment_ecg_image(image)
    Image.fromarray(aug_img).save(f'./pngs/augmented_ecg_{i}.png')

