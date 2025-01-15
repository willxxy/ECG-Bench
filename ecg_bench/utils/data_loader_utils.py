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
        
        # For datasets that don't have question
        self.uniform_question = 'Could you please help me explain my ECG?'
        
    def __len__(self):
        return len(self.json_data_file)

    def __getitem__(self, idx):
        instance = self.json_data_file[idx]
        self.prepare_instance(instance)
        
        return_dic = {

        }
        
        return return_dic
        
    def prepare_instance(self, instance):
        ### from json file
        altered_text = instance['text']
        if self.args.data == 'pretrain_mimic_mapped':
            question, answer = altered_text[0]['value'].replace('\n', '').replace('<ecg>', ''), altered_text[1]['value']
        elif self.args.data in ['ecg-qa_mimic-iv-ecg_mapped', 'ecg-qa_ptbxl_mapped']:
            question_type, question, answer = altered_text[0], altered_text[1], altered_text[2]
            answer = ' '.join(answer) if isinstance(answer, list) else answer
        print(question)
        print(answer)
        
        ### from numpy file
        np_path = instance['ecg_path']
        ecg_path = self.fm.open_npy(np_path)
        ecg_signal = ecg_path['ecg']
        original_report = ecg_path['report']
        print(ecg_signal.shape)
        print(original_report)
        
        return_dic = {
            'question': question,
            'answer': answer,
            'ecg': ecg_signal,
            'original_report': original_report
        }
        