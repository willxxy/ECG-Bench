import numpy as np
from torch.utils.data import Dataset
import torch
import json
from PIL import Image


class ECGDataset(Dataset):
    def __init__(self, json_data_file, 
                 fm, args, viz, tokenizer_utils,
                 encoder_tokenizer = None, encoder_tokenizer2 = None):
        self.json_data_file = json_data_file
        self.fm = fm
        self.args = args
        self.viz = viz
        self.tokenizer_utils = tokenizer_utils
        self.encoder_tokenizer = encoder_tokenizer
        self.encoder_tokenizer2 = encoder_tokenizer2
        
        # For datasets that don't have question
        self.uniform_question = 'Could you please help me explain my ECG?'
        
    def __len__(self):
        return len(self.json_data_file)

    def __getitem__(self, idx):
        instance = self.json_data_file[idx]
        np_path = instance['ecg_path']
        ecg_path = self.fm.open_npy(np_path)
        ecg_signal = ecg_path['ecg']
        original_report = ecg_path['report']
        altered_text = instance['text']
        
        if self.args.model == 'clip':
            return_dict = self.prepare_clip_input(ecg_signal, original_report)
        
        return return_dict
    
    def prepare_clip_input(self, ecg_signal, original_report):
        # self.viz.plot_2d_ecg(ecg_signal, 'ecg_signal', save_path = './pngs/', sample_rate = 250)
        # print('ecg_signal:', ecg_signal.shape)
        normalized_signal, _ = self.tokenizer_utils.normalize(ecg_signal)
        rgb_norm_signal = np.stack([normalized_signal * 255] * 3, axis = -1).astype(np.uint8)
        image_signal = Image.fromarray(rgb_norm_signal)
        # print('normalized_signal:', normalized_signal.shape)
        # self.viz.plot_2d_ecg(normalized_signal, 'normalized_signal', save_path = './pngs/', sample_rate = 250)
        clip_inputs = self.encoder_tokenizer(text = [original_report],
                                             images = [image_signal],
                                             return_tensors = 'pt',
                                             padding = 'max_length',
                                             max_length = 77,
                                             truncation = True)
        input_ids = clip_inputs['input_ids'][0]
        attention_mask = clip_inputs['attention_mask'][0]
        pixel_values = clip_inputs['pixel_values'][0]
        return {
            'clip_input_ids': input_ids,
            'clip_att_mask': attention_mask,
            'clip_pixel': pixel_values
        }

        
        