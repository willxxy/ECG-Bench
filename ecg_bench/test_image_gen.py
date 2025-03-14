import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../ecg-image-kit/codes/ecg-image-generator"))
from gen_ecg_image_from_np import generate_ecg_image
import glob
path_to_npy = glob.glob('./data/mimic/preprocessed_1250_250/*.npy')[0]

test_file = np.load(path_to_npy, allow_pickle = True).item()
print(test_file.keys())
ecg = test_file['ecg']
print(ecg.shape)

lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

from utils.viz_utils import VizUtil
VizUtil.plot_2d_ecg(ecg, '', 'test', 250)

image = VizUtil.get_plot_as_image(ecg, 250)
print(image)
print(image.shape)

from PIL import Image
img = Image.fromarray(image)
img.save('test_from_array.png')