import numpy as np
from torch.utils.data import Dataset
import torch
import json
from PIL import Image


class ECGDataset(Dataset):
    def __init__(self, json_data_file, fm, args):
        self.json_data_file = json_data_file
        self.fm = fm
        self.args = args
        
    def __len__(self):
        return len(self.json_data_file)

    def __getitem__(self, idx):
        instance = self.json_data_file[idx]
        ecg_path = instance['ecg_path']
        ecg_signal = self.fm.open_ecg(ecg_path)
        text = instance['text']
        
        return_dic = {
            'ecg_path': ecg_path,
            'text': text
        }
        
        return return_dic
        