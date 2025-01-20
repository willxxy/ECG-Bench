import numpy as np
from torch.utils.data import Dataset
import torch
import json
from PIL import Image


class ECGDataset(Dataset):
    def __init__(self, json_data_file, 
                 args, train_utils, encoder_tokenizer = None, 
                 encoder_tokenizer2 = None, llm_tokenizer = None):
        self.json_data_file = json_data_file
        self.train_utils = train_utils
        self.args = args
    
        self.encoder_tokenizer = encoder_tokenizer
        self.encoder_tokenizer2 = encoder_tokenizer2
        self.llm_tokenizer = llm_tokenizer
        if llm_tokenizer != None:
            self.create_special_tokens()
        # For datasets that don't have question
        self.uniform_question = 'Could you please help me explain my ECG?'
        
    def __len__(self):
        return len(self.json_data_file)

    def __getitem__(self, idx):
        try:
            instance = self.json_data_file[idx]
            np_path = instance['ecg_path']
            ecg_path = self.train_utils.fm.open_npy(np_path)
            ecg_signal = ecg_path['ecg']
            original_report = ecg_path['report']
            altered_text = instance['text']
            
            if self.args.model == 'clip':
                return self.prepare_clip_input(ecg_signal, original_report)
            elif self.args.model == 'vit':
                return self.prepare_vit_input(ecg_signal)
            elif self.args.model == 'llama-3.2-1b':
                return self.prepare_end2end_input(ecg_signal, altered_text)
            elif self.args.model == 'merl':
                return self.prepare_merl_input(ecg_signal, original_report)
            
        except Exception as e:
            print(e)
            print(f"Skipping invalid data at index {idx}")
            return None
    
    def prepare_inference_end2end(self, tokenized_signal, tokenized_question, answer, question):
        inference_seq = [self.bos_id] + self.sig_start_id + tokenized_signal + self.sig_end_id + tokenized_question
        attention_mask = self.create_attention_mask(self.pad_id, inference_seq)
        return {
            'answer': answer,
            'question': question,
            'input_ids': torch.tensor(inference_seq, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32)
        }

    def prepare_training_end2end(self, tokenized_signal, tokenized_question, tokenized_answer, signal):

        qa_len = len(tokenized_question) + len(tokenized_answer)
        available_space = self.args.pad_to_max - qa_len

        ### We dont use pad_to_max since we only want to pad/truncate the signal not text
        if len(tokenized_signal) > available_space:
            tokenized_signal = [self.bos_id] + self.sig_start_id + tokenized_signal[:available_space] + self.sig_end_id
        elif len(tokenized_signal) < available_space:
            tokenized_signal = [self.pad_id] * (available_space - len(tokenized_signal)) + [self.bos_id] + self.sig_start_id + tokenized_signal + self.sig_end_id
        else:
            tokenized_signal = [self.bos_id] + self.sig_start_id + tokenized_signal + self.sig_end_id

        input_ids = tokenized_signal + tokenized_question + tokenized_answer + [self.eos_id]
        labels = [-100] * (len(tokenized_signal) + len(tokenized_question)) + tokenized_answer + [self.eos_id]
        labels = torch.tensor(labels, dtype=torch.int64)
        position_ids = self.create_position_ids(input_ids)
        attention_mask = self.create_attention_mask(input_ids)

        assert len(input_ids) == len(attention_mask) == (self.args.pad_to_max + 4) == labels.shape[0] == position_ids.shape[0], \
            f"Lengths don't match: masked_sample ({len(input_ids)}), attention_mask ({len(attention_mask)})"

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'labels': labels,
            'position_ids': position_ids,
            'signal': signal,
        }
    
    def create_special_tokens(self):
        self.pad_id = self.llm_tokenizer.convert_tokens_to_ids(self.llm_tokenizer.pad_token)
        self.bos_id = self.llm_tokenizer.convert_tokens_to_ids(self.llm_tokenizer.bos_token)
        self.eos_id = self.llm_tokenizer.convert_tokens_to_ids(self.llm_tokenizer.eos_token)
        self.sig_start_id = self.llm_tokenizer.convert_tokens_to_ids(['<sig_start>'])
        self.sig_end_id = self.llm_tokenizer.convert_tokens_to_ids(['<sig_end>'])
        if self.args.train == 'second' or self.args.inference == 'second':
            self.signal_id = self.llm_tokenizer.convert_tokens_to_ids(['<signal>'])
    
    def prepare_training_second(self, encoder_out, tokenized_question, tokenized_answer):
        ### Don't need to add eos or bos id since we do that in pad_to_max
        tokenized_signal = self.sig_start_id + self.signal_id + self.sig_end_id + tokenized_question + tokenized_answer
        labels = ([-100] * (3 + len(tokenized_question))) + tokenized_answer
        input_ids = self.pad_to_max(tokenized_signal)        
        signal_id_index = input_ids.index(self.signal_id[0])  # [0] because signal_id is a list
        labels = self.pad_to_max(labels)
        labels[labels == self.pad_id] = -100
        labels[labels == self.bos_id] = -100
        attention_mask = self.create_attention_mask(input_ids)
        position_ids = self.create_position_ids(input_ids)
        
        assert len(input_ids) == len(attention_mask) == (self.args.pad_to_max + 2) == labels.shape[0] == position_ids.shape[0], \
            f"Lengths don't match: masked_sample ({len(input_ids)}), attention_mask ({len(attention_mask)})"
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'position_ids': position_ids,
            'encoder_out': encoder_out,
            'signal_id_index': signal_id_index
        }
    
    def prepare_inference_second(self, encoder_out, tokenized_question, tokenized_answer):
        pass
    
    def prepare_second_input(self, ecg_signal, altered_text, original_report = None):
        question, answer = self.get_qa(altered_text)
        tokenized_question = self.llm_tokenizer([question], return_tensors = 'np', add_special_tokens = False).input_ids[0].tolist()
        tokenized_answer = self.llm_tokenizer([answer], return_tensors = 'np', add_special_tokens = False).input_ids[0].tolist()
        if 'vit' in self.args.model:
            encoder_out = self.prepare_vit_input(ecg_signal)
        elif 'clip' in self.args.model:
            encoder_out = self.prepare_clip_input(ecg_signal, original_report)
        elif 'merl' in self.args.model:
            encoder_out = self.prepare_merl_input(ecg_signal, original_report)
        
        if self.args.train == 'second' and self.args.inference == None:
            return self.prepare_training_second(encoder_out, tokenized_question, tokenized_answer)
        if self.args.inference == 'second' and self.args.train == None:
            return self.prepare_inference_second(encoder_out, tokenized_question, tokenized_answer)
        
        
    def prepare_end2end_input(self, ecg_signal, altered_text, original_report = None):
        question, answer = self.get_qa(altered_text)
        symbol_signal = self.train_utils.ecg_tokenizer_utils._to_symbol_string(ecg_signal)
        encoded_signal = self.train_utils.ecg_tokenizer_utils.encode_symbol(symbol_signal, 
                                                                            self.train_utils.ecg_tokenizer_utils.merges)
        tokenized_signal = self.llm_tokenizer.convert_tokens_to_ids([f'signal_{ids}' for ids in encoded_signal])
        tokenized_question = self.llm_tokenizer([question], return_tensors = 'np', add_special_tokens = False).input_ids[0].tolist()
        tokenized_answer = self.llm_tokenizer([answer], return_tensors = 'np', add_special_tokens = False).input_ids[0].tolist()
        
        if self.args.train == 'end2end' and self.args.inference == None:
            return self.prepare_training_end2end(tokenized_signal, tokenized_question, tokenized_answer, ecg_signal)
        if self.args.inference == 'end2end' and self.args.train == None:
            return self.prepare_inference_end2end(tokenized_signal, tokenized_question, answer, question)
                
    def prepare_clip_input(self, ecg_signal, original_report):
        image_signal = self.signal_to_image(ecg_signal)
        clip_inputs = self.encoder_tokenizer(text = [original_report],
                                             images = [image_signal],
                                             return_tensors = 'pt',
                                             padding = 'max_length',
                                             max_length = 77,
                                             truncation = True)
        input_ids = clip_inputs['input_ids'][0].contiguous()
        attention_mask = clip_inputs['attention_mask'][0].contiguous()
        pixel_values = clip_inputs['pixel_values'][0].contiguous()
        return {
            'clip_input_ids': input_ids,
            'clip_att_mask': attention_mask,
            'clip_pixel': pixel_values
        }
    
    def prepare_merl_input(self, ecg_signal, original_report):
        normalized_signal, _ = self.train_utils.ecg_tokenizer_utils.normalize(ecg_signal)
        merl_inputs = self.encoder_tokenizer(text = [original_report],
                                             return_tensors = 'pt',
                                             padding = 'max_length',
                                             max_length = 64,
                                             truncation = True)
        input_ids = merl_inputs['input_ids'][0].contiguous()
        attention_mask = merl_inputs['attention_mask'][0].contiguous()
        return {
                'merl_input_ids': input_ids,
                'merl_att_mask': attention_mask,
                'signal': normalized_signal.astype(np.float32)
                }
    
    def prepare_vit_input(self, ecg_signal):
        image_signal = self.signal_to_image(ecg_signal)
        vit_inputs = self.encoder_tokenizer(images = image_signal,
                                             return_tensors = 'pt')
        pixel_values = vit_inputs['pixel_values'][0].contiguous()
        mask = torch.rand(size=(1, self.args.num_patches)) < 0.75
        mask = mask[0].contiguous()
        return {
            'vit_pixel': pixel_values,
            'vit_mask': mask
        }
    
    def signal_to_image(self, signal):
        normalized_signal, _ = self.train_utils.ecg_tokenizer_utils.normalize(signal)
        rgb_norm_signal = np.stack([normalized_signal * 255] * 3, axis = -1).astype(np.uint8)
        return Image.fromarray(rgb_norm_signal)

    def create_attention_mask(self, numbers):
        return [0 if num == self.pad_id else 1 for num in numbers]

    def create_position_ids(self, padded_sequence):
        padded_sequence = torch.tensor(padded_sequence)
        mask = (padded_sequence != self.pad_id).long()
        position_ids = torch.cumsum(mask, dim=0) - 1
        position_ids.masked_fill_(mask == 0, 0)
        return position_ids
    
    def get_qa(self, altered_text):
        if self.args.data == 'pretrain_mimic_mapped':
            question, answer = altered_text[0]['value'].replace('\n', '').replace('<ecg>', ''), altered_text[1]['value']
        elif self.args.data in ['ecg-qa_mimic-iv-ecg_mapped', 'ecg-qa_ptbxl_mapped']:
            question_type, question, answer = altered_text[0], altered_text[1], altered_text[2]
            answer = ' '.join(answer) if isinstance(answer, list) else answer
        return question, answer
    
    def pad_to_max(self, tokenized_sequence):
        if len(tokenized_sequence) > self.args.pad_to_max:
            truncated_token = tokenized_sequence[:self.args.pad_to_max]
            full_token = [self.bos_id] + list(truncated_token) + [self.eos_id]
            return full_token
        elif len(tokenized_sequence) < self.args.pad_to_max:
            return [self.pad_id] * (self.args.pad_to_max - len(tokenized_sequence)) + [self.bos_id] + list(tokenized_sequence) + [self.eos_id]
        else:
            return [self.bos_id] + list(tokenized_sequence[:self.args.pad_to_max]) + [self.eos_id]