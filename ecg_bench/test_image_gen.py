import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../ecg-image-kit/codes/ecg-image-generator"))
from gen_ecg_image_from_np import generate_ecg_image

test_file = np.load('./data/ptb/preprocessed_500_250/records500_09000_09537_hr_1.npy', allow_pickle = True).item()
print(test_file.keys())
ecg = test_file['ecg']
print(ecg.shape)

lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

from utils.viz_utils import VizUtil
VizUtil.plot_2d_ecg(ecg, '', 'test', 250)