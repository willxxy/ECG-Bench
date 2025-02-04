import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image


class BaseECGDataset(Dataset):
    def __init__(self, json_data_file, train_utils, encoder_tokenizer=None, llm_tokenizer=None):
        self.json_data_file = json_data_file
        self.train_utils = train_utils
        self.args = self.train_utils.args
        self.encoder_tokenizer = encoder_tokenizer
        self.llm_tokenizer = llm_tokenizer
        if llm_tokenizer is not None:
            self.create_special_tokens()
        self.uniform_question = 'Could you please help me explain my ECG?'
    
    def __len__(self):
        return len(self.json_data_file)

    def signal_to_image(self, signal):
        normalized_signal, _ = self.train_utils.ecg_tokenizer_utils.normalize(signal)
        rgb_norm_signal = np.stack([normalized_signal * 255] * 3, axis=-1).astype(np.uint8)
        return Image.fromarray(rgb_norm_signal)

    def create_attention_mask(self, input_ids):
        return [0 if num == self.pad_id else 1 for num in input_ids]

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
        
    def create_special_tokens(self):
        self.pad_id = self.llm_tokenizer.convert_tokens_to_ids(self.llm_tokenizer.pad_token)
        self.bos_id = self.llm_tokenizer.convert_tokens_to_ids(self.llm_tokenizer.bos_token)
        self.eos_id = self.llm_tokenizer.convert_tokens_to_ids(self.llm_tokenizer.eos_token)
        self.sig_start_id = self.llm_tokenizer.convert_tokens_to_ids(['<sig_start>'])
        self.sig_end_id = self.llm_tokenizer.convert_tokens_to_ids(['<sig_end>'])


class EncoderInputPreparation(BaseECGDataset):
    def __init__(self, encoder_tokenizer, train_utils):
        super().__init__(json_data_file=None, train_utils=train_utils, encoder_tokenizer=encoder_tokenizer)

    def prepare_vit_input(self, ecg_signal, num_patches):
        image_signal = self.signal_to_image(ecg_signal)
        vit_inputs = self.encoder_tokenizer(images=image_signal,
                                          return_tensors='pt')
        pixel_values = vit_inputs['pixel_values'][0].contiguous()
        mask = torch.rand(size=(1, num_patches)) < 0.75
        return {
            'vit_pixel': pixel_values,
            'vit_mask': mask[0].contiguous()
        }
    
    def prepare_clip_input(self, ecg_signal, original_report):
        image_signal = self.signal_to_image(ecg_signal)
        clip_inputs = self.encoder_tokenizer(text=[original_report],
                                           images=[image_signal],
                                           return_tensors='pt',
                                           padding='max_length',
                                           max_length=77,
                                           truncation=True)
        return {
            'clip_input_ids': clip_inputs['input_ids'][0].contiguous(),
            'clip_att_mask': clip_inputs['attention_mask'][0].contiguous(),
            'clip_pixel': clip_inputs['pixel_values'][0].contiguous()
        }
        
    def prepare_siglip_input(self, ecg_signal, original_report):
        image_signal = self.signal_to_image(ecg_signal)
        clip_inputs = self.encoder_tokenizer(text=[original_report],
                                           images=[image_signal],
                                           return_tensors='pt',
                                           padding='max_length',
                                           max_length=64,
                                           truncation=True)
        return {
            'siglip_input_ids': clip_inputs['input_ids'][0].contiguous(),
            'siglip_att_mask': clip_inputs['attention_mask'][0].contiguous(),
            'siglip_pixel': clip_inputs['pixel_values'][0].contiguous()
        }
    
    def prepare_merl_input(self, ecg_signal, original_report):
        normalized_signal, _ = self.train_utils.ecg_tokenizer_utils.normalize(ecg_signal)
        merl_inputs = self.encoder_tokenizer(text=[original_report],
                                           return_tensors='pt',
                                           padding='max_length',
                                           max_length=64,
                                           truncation=True)
        return {
            'merl_input_ids': merl_inputs['input_ids'][0].contiguous(),
            'merl_att_mask': merl_inputs['attention_mask'][0].contiguous(),
            'signal': normalized_signal.astype(np.float32)
        }


class FirstStageECGDataset(BaseECGDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_prep = EncoderInputPreparation(self.encoder_tokenizer, self.train_utils)

    def __getitem__(self, idx):
        try:
            instance = self.json_data_file[idx]
            np_path = instance['ecg_path']
            ecg_path = self.train_utils.fm.open_npy(np_path)
            ecg_signal = ecg_path['ecg']
            original_report = ecg_path['report']
            
            if 'clip' in self.args.model:
                return self.encoder_prep.prepare_clip_input(ecg_signal, original_report)
            elif 'vit' in self.args.model:
                return self.encoder_prep.prepare_vit_input(ecg_signal, self.args.num_patches)
            elif 'merl' in self.args.model:
                return self.encoder_prep.prepare_merl_input(ecg_signal, original_report)
        except Exception as e:
            print(e)
            print(f"Skipping invalid data at index {idx}")
            return None


class SecondStageECGDataset(BaseECGDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.llm_tokenizer is not None:
            self.signal_id = self.llm_tokenizer.convert_tokens_to_ids(['<signal>'])
        self.encoder_prep = EncoderInputPreparation(self.encoder_tokenizer, self.train_utils)

    def __getitem__(self, idx):
        try:
            instance = self.json_data_file[idx]
            np_path = instance['ecg_path']
            ecg_path = self.train_utils.fm.open_npy(np_path)
            ecg_signal = ecg_path['ecg']
            original_report = ecg_path['report']
            altered_text = instance['text']
            
            return self.prepare_second_input(ecg_signal, altered_text, original_report)
        except Exception as e:
            print(e)
            print(f"Skipping invalid data at index {idx}")
            return None

    def prepare_second_input(self, ecg_signal, altered_text, original_report=None):
        question, answer = self.get_qa(altered_text)
        tokenized_question = self.llm_tokenizer([question], return_tensors='np', add_special_tokens=False).input_ids[0].tolist()
        tokenized_answer = self.llm_tokenizer([answer], return_tensors='np', add_special_tokens=False).input_ids[0].tolist()
        
        if 'vit' in self.args.model:
            encoder_out = self.encoder_prep.prepare_vit_input(ecg_signal, self.args.num_patches)
        elif 'clip' in self.args.model:
            encoder_out = self.encoder_prep.prepare_clip_input(ecg_signal, original_report)
        elif 'merl' in self.args.model:
            encoder_out = self.encoder_prep.prepare_merl_input(ecg_signal, original_report)
        
        if self.args.train == 'second' and self.args.inference is None:
            return self.prepare_training_second(encoder_out, tokenized_question, tokenized_answer)
        if self.args.inference == 'second' and self.args.train is None:
            return self.prepare_inference_second(encoder_out, tokenized_question, answer, question)

    def prepare_training_second(self, encoder_out, tokenized_question, tokenized_answer):
        input_ids = self.sig_start_id + self.signal_id + self.sig_end_id + tokenized_question + tokenized_answer
        labels = ([-100] * (3 + len(tokenized_question))) + tokenized_answer
        input_ids = self.pad_to_max(input_ids)        
        signal_id_index = input_ids.index(self.signal_id[0])
        
        labels = self.pad_to_max(labels)
        labels[labels == self.pad_id] = -100
        labels[labels == self.bos_id] = -100
        labels = torch.tensor(labels, dtype=torch.int64)
        attention_mask = self.create_attention_mask(input_ids)
        position_ids = self.create_position_ids(input_ids)
        
        assert len(input_ids) == len(attention_mask) == (self.args.pad_to_max + 2) == labels.shape[0] == position_ids.shape[0]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'labels': labels,
            'position_ids': position_ids,
            'encoder_out': encoder_out,
            'signal_id_index': signal_id_index
        }
    
    def prepare_inference_second(self, encoder_out, tokenized_question, answer, question):
        input_ids = [self.bos_id] + self.sig_start_id + self.signal_id + self.sig_end_id + tokenized_question
        signal_id_index = input_ids.index(self.signal_id[0])
        attention_mask = self.create_attention_mask(input_ids)
        return {
            'answer': answer,
            'question': question,
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'signal_id_index': signal_id_index,
            'encoder_out': encoder_out
        }


class End2EndECGDataset(BaseECGDataset):
    def __getitem__(self, idx):
        try:
            instance = self.json_data_file[idx]
            np_path = instance['ecg_path']
            ecg_path = self.train_utils.fm.open_npy(np_path)
            ecg_signal = ecg_path['ecg']
            altered_text = instance['text']
            
            return self.prepare_end2end_input(ecg_signal, altered_text)
        except Exception as e:
            print(e)
            print(f"Skipping invalid data at index {idx}")
            return None

    def prepare_end2end_input(self, ecg_signal, altered_text):
        question, answer = self.get_qa(altered_text)
        symbol_signal = self.train_utils.ecg_tokenizer_utils._to_symbol_string(ecg_signal)
        encoded_signal = self.train_utils.ecg_tokenizer_utils.encode_symbol(symbol_signal, 
                                                                          self.train_utils.ecg_tokenizer_utils.merges)
        tokenized_signal = self.llm_tokenizer.convert_tokens_to_ids([f'signal_{ids}' for ids in encoded_signal])
        tokenized_question = self.llm_tokenizer([question], return_tensors='np', add_special_tokens=False).input_ids[0].tolist()
        tokenized_answer = self.llm_tokenizer([answer], return_tensors='np', add_special_tokens=False).input_ids[0].tolist()
        
        if self.args.train == 'end2end' and self.args.inference is None:
            return self.prepare_training_end2end(tokenized_signal, tokenized_question, tokenized_answer, ecg_signal)
        if self.args.inference == 'end2end' and self.args.train is None:
            return self.prepare_inference_end2end(tokenized_signal, tokenized_question, answer, question)

    def prepare_training_end2end(self, tokenized_signal, tokenized_question, tokenized_answer, signal):
        qa_len = len(tokenized_question) + len(tokenized_answer)
        available_space = self.args.pad_to_max - qa_len

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

        assert len(input_ids) == len(attention_mask) == (self.args.pad_to_max + 4) == labels.shape[0] == position_ids.shape[0]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'labels': labels,
            'position_ids': position_ids,
            'signal': signal,
        }

    def prepare_inference_end2end(self, tokenized_signal, tokenized_question, answer, question):
        input_ids = [self.bos_id] + self.sig_start_id + tokenized_signal + self.sig_end_id + tokenized_question
        attention_mask = self.create_attention_mask(input_ids)
        return {
            'answer': answer,
            'question': question,
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32)
        }