import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
from ecg_bench.utils.conversation_utils import get_conv_template
from imgaug import augmenters as iaa
import random

class BaseECGDataset(Dataset):
    def __init__(self, json_data_file, train_utils, encoder_tokenizer=None, llm_tokenizer=None):
        self.json_data_file = json_data_file
        self.train_utils = train_utils
        self.viz = train_utils.viz
        self.args = self.train_utils.args
        self.encoder_tokenizer = encoder_tokenizer
        self.llm_tokenizer = llm_tokenizer
        if llm_tokenizer is not None:
            self.create_special_tokens()
        if self.args.train == 'end2end' or self.args.inference == 'end2end' or self.args.train == 'second' or self.args.inference == 'second':
            self.system_prompt = self.train_utils.fm.get_system_prompt(self.args.system_prompt)
            self.ecg_placeholder = '<signal>'
        self.uniform_question = 'Could you please help me explain my ECG?'
    
    def __len__(self):
        return len(self.json_data_file)

    def signal_to_image(self, signal):
        if self.args.image:
            image = self.viz.get_plot_as_image(signal, self.args.target_sf)
            if random.random() < 0.6:
                return self.augment_image(image)
            else:
                return Image.fromarray(image)
        else:
            if self.args.instance_normalize:
                normalized_signal, _ = self.train_utils.ecg_tokenizer_utils.instance_normalize(signal)
            else:
                normalized_signal, _ = self.train_utils.ecg_tokenizer_utils.normalize(signal)
            rgb_norm_signal = np.stack([normalized_signal * 255] * 3, axis=-1).astype(np.uint8)
            return Image.fromarray(rgb_norm_signal)
    
    def augment_image(self, image):
        seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),    # 50% chance to change brightness
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-5, 5))),    # 50% chance to rotate by -5° to 5°
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 1.5))),  # 50% chance to apply Gaussian blur
        iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-30, 30), per_channel=True))  # 50% chance to adjust hue
        ])
        augmented_image = seq.augment_image(image)
        return Image.fromarray(augmented_image)

    def create_attention_mask(self, input_ids):
        return [0 if num == self.pad_id else 1 for num in input_ids]

    def create_position_ids(self, padded_sequence):
        padded_sequence = torch.tensor(padded_sequence)
        mask = (padded_sequence != self.pad_id).long()
        position_ids = torch.cumsum(mask, dim=0) - 1
        position_ids.masked_fill_(mask == 0, 0)
        return position_ids
    
    def get_qa(self, altered_text):
        if self.args.data == f'pretrain_mimic_mapped_{self.args.seg_len}':
            question, answer = altered_text[0]['value'].replace('\n', '').replace('<ecg>', ''), altered_text[1]['value']
        elif self.args.data in [f'ecg-qa_mimic-iv-ecg_mapped_{self.args.seg_len}', f'ecg-qa_ptbxl_mapped_{self.args.seg_len}']:
            question_type, question, answer = altered_text[0], altered_text[1], altered_text[2]
            answer = ' '.join(answer) if isinstance(answer, list) else answer
        return question, answer
    
    def pad_to_max_qa(self, tokenized_sequence):
        if len(tokenized_sequence) > self.args.pad_to_max:
            truncated_token = tokenized_sequence[:self.args.pad_to_max]
            full_token = [self.bos_id] + list(truncated_token) + [self.eos_id]
            return full_token
        elif len(tokenized_sequence) < self.args.pad_to_max:
            return [self.pad_id] * (self.args.pad_to_max - len(tokenized_sequence)) + [self.bos_id] + list(tokenized_sequence) + [self.eos_id]
        else:
            return [self.bos_id] + list(tokenized_sequence[:self.args.pad_to_max]) + [self.eos_id]
        
    def pad_to_max_chat(self, tokenized_sequence):
        if len(tokenized_sequence) > self.args.pad_to_max:
            truncated_token = tokenized_sequence[:self.args.pad_to_max]
            return list(truncated_token)
        elif len(tokenized_sequence) < self.args.pad_to_max:
            return [self.pad_id] * (self.args.pad_to_max - len(tokenized_sequence)) + list(tokenized_sequence)
        else:
            return list(tokenized_sequence[:self.args.pad_to_max]) # explicitly return truncated sequence in the case where the sequence is exactly the max length
        
    def create_special_tokens(self):
        self.pad_id = self.llm_tokenizer.convert_tokens_to_ids(self.llm_tokenizer.pad_token)


class EncoderInputPreparation(BaseECGDataset):
    def __init__(self, encoder_tokenizer, train_utils):
        super().__init__(json_data_file=None, train_utils=train_utils, encoder_tokenizer=encoder_tokenizer)

    def prepare_st_mem_input(self, ecg_signal):
        normalized_signal, _ = self.train_utils.ecg_tokenizer_utils.normalize(ecg_signal)
        return {'signal': normalized_signal.astype(np.float32)}
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
        siglip_inputs = self.encoder_tokenizer(text=[original_report],
                                           images=[image_signal],
                                           return_tensors='pt',
                                           padding='max_length',
                                           max_length=64,
                                           truncation=True)
        ### siglip does not have attention mask??
        pad_token_id = 1 # so we define the pad token manually (This is the id for pad in siglip)
        attention_mask = (siglip_inputs['input_ids'][0] != pad_token_id).int()
        return {
            'siglip_input_ids': siglip_inputs['input_ids'][0].contiguous(),
            'siglip_att_mask': attention_mask.contiguous(),
            'siglip_pixel': siglip_inputs['pixel_values'][0].contiguous()
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
            elif 'siglip' in self.args.model:
                return self.encoder_prep.prepare_siglip_input(ecg_signal, original_report)
            elif 'merl' in self.args.model:
                return self.encoder_prep.prepare_merl_input(ecg_signal, original_report)
            elif 'stmem' in self.args.model:
                return self.encoder_prep.prepare_st_mem_input(ecg_signal)
            elif 'mtae' in self.args.model:
                return self.encoder_prep.prepare_st_mem_input(ecg_signal)
            elif 'mlae' in self.args.model:
                return self.encoder_prep.prepare_st_mem_input(ecg_signal)
        except Exception as e:
            print(e)
            print(f"Skipping invalid data at index {idx}")
            return None

        
class End2EndECGChatDataset(BaseECGDataset):
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
        if self.args.train == 'end2end' and self.args.inference is None:
            return self.prepare_training_end2end(ecg_signal, altered_text)
        if self.args.inference == 'end2end' and self.args.train is None:
            return self.prepare_inference_end2end(ecg_signal, altered_text)
    
    def prepare_training_end2end(self, ecg_signal, altered_text):
        if 'llama' in self.args.model:
            conv = get_conv_template('llama-3')
        conv.set_system_message(self.system_prompt)
        
        if self.args.data not in [f'ecg_instruct_45k_mapped_{self.args.seg_len}', 
                                  f'ecg_instruct_pulse_mapped_{self.args.seg_len}',
                                  f'ecg_bench_pulse_mapped_{self.args.seg_len}']:
            question, answer = self.get_qa(altered_text)
            altered_text = [{'from': 'human', 'value': question}, {'from': 'assistant', 'value': answer}]
        
        count = 0
        for message in altered_text:
            is_human = message['from'].lower() in ['human', 'user']
            role = conv.roles[0] if is_human else conv.roles[1]
            message_value = message['value'].replace('<ecg>\n', '')
            message_value = message_value.replace('<image>\n', '')
            message_value = message_value.replace('image', 'signal').replace('Image', 'Signal')
            if is_human and count == 0:
                message_value = f"<signal>\n{message_value}"
                count += 1
            conv.append_message(role, message_value)
            
        prompt = conv.get_prompt()
        ecg_position = prompt.find(self.ecg_placeholder)
        prompt_before_ecg = prompt[:ecg_position]
        prompt_after_ecg = prompt[ecg_position + len(self.ecg_placeholder):]
        tokens_before = self.llm_tokenizer.encode(prompt_before_ecg, add_special_tokens=False)
        tokens_after = self.llm_tokenizer.encode(prompt_after_ecg, add_special_tokens=False)
        
        symbol_signal = self.train_utils.ecg_tokenizer_utils._to_symbol_string(ecg_signal)
        encoded_signal = self.train_utils.ecg_tokenizer_utils.encode_symbol(symbol_signal, 
                                                                          self.train_utils.ecg_tokenizer_utils.merges)
        tokenized_signal = self.llm_tokenizer.convert_tokens_to_ids([f'signal_{ids}' for ids in encoded_signal])
        signal_tokens = tokenized_signal[:min(500, len(tokenized_signal))]
        max_conv_tokens = self.args.pad_to_max - len(signal_tokens)

        if len(tokens_before) + len(tokens_after) > max_conv_tokens:
            max_after = max(max_conv_tokens // 3, 1)
            tokens_after = tokens_after[:max_after]
            tokens_before = tokens_before[-(max_conv_tokens - len(tokens_after)):]
        
        total_used = len(tokens_before) + len(tokens_after) + len(signal_tokens)
        if total_used < self.args.pad_to_max and len(signal_tokens) < len(tokenized_signal):
            extra_signal = min(self.args.pad_to_max - total_used, len(tokenized_signal) - len(signal_tokens))
            signal_tokens = tokenized_signal[:len(signal_tokens) + extra_signal]
        
        input_ids = tokens_before + signal_tokens + tokens_after
        
        if len(input_ids) < self.args.pad_to_max:
            padding_length = self.args.pad_to_max - len(input_ids)
            input_ids = [self.llm_tokenizer.pad_token_id] * padding_length + input_ids
        
        labels = [-100] * len(input_ids)
        for message in altered_text:
            if message['from'].lower() == 'assistant':
                response = message['value']
                response_tokens = self.llm_tokenizer.encode(response, add_special_tokens=False)
                for j in range(len(input_ids) - len(response_tokens) + 1):
                    if input_ids[j:j+len(response_tokens)] == response_tokens:
                        labels[j:j+len(response_tokens)] = response_tokens
        
        eot_id = self.llm_tokenizer.convert_tokens_to_ids('<|eot_id|>')
        for i, token_id in enumerate(input_ids):
            if token_id == eot_id:
                labels[i] = eot_id
        
        
        assert len(input_ids) == self.args.pad_to_max, f"Expected length {self.args.pad_to_max}, got {len(input_ids)}"
        
        labels = torch.tensor(labels, dtype=torch.int64)    
        position_ids = self.create_position_ids(input_ids)
        attention_mask = self.create_attention_mask(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'labels': labels,
            'position_ids': position_ids,
            'signal': ecg_signal,
        }
    
    def prepare_inference_end2end(self, ecg_signal, altered_text):
        if 'llama' in self.args.model:
            conv = get_conv_template('llama-3')
        conv.set_system_message(self.system_prompt)
        
        if self.args.data not in [f'ecg_instruct_45k_mapped_{self.args.seg_len}', 
                                  f'ecg_instruct_pulse_mapped_{self.args.seg_len}',
                                  f'ecg_bench_pulse_mapped_{self.args.seg_len}']:
            question, answer = self.get_qa(altered_text)
            altered_text = [{'from': 'human', 'value': question}, {'from': 'assistant', 'value': answer}]
        
        count = 0
        for message in altered_text:
            is_human = message['from'].lower() in ['human', 'user']
            role = conv.roles[0] if is_human else conv.roles[1]
            message_value = message['value'].replace('<ecg>\n', '')
            message_value = message_value.replace('<image>\n', '')
            message_value = message_value.replace('image', 'signal').replace('Image', 'Signal')
            if is_human and count == 0:
                message_value = f"<signal>\n{message_value}"
                count += 1
            conv.append_message(role, message_value)
            
        prompt = conv.get_prompt()
        ecg_position = prompt.find(self.ecg_placeholder)
        prompt_before_ecg = prompt[:ecg_position]
        prompt_after_ecg = prompt[ecg_position + len(self.ecg_placeholder):]
        tokens_before = self.llm_tokenizer.encode(prompt_before_ecg, add_special_tokens=False)
        tokens_after = self.llm_tokenizer.encode(prompt_after_ecg, add_special_tokens=False)
                
        symbol_signal = self.train_utils.ecg_tokenizer_utils._to_symbol_string(ecg_signal)
        encoded_signal = self.train_utils.ecg_tokenizer_utils.encode_symbol(symbol_signal, 
                                                                          self.train_utils.ecg_tokenizer_utils.merges)
        tokenized_signal = self.llm_tokenizer.convert_tokens_to_ids([f'signal_{ids}' for ids in encoded_signal])
        input_ids = tokens_before + tokenized_signal + tokens_after
        attention_mask = self.create_attention_mask(input_ids)
        
        assistant_ranges = []
        start_header_id = self.llm_tokenizer.convert_tokens_to_ids(['<|start_header_id|>'])[0]
        assistant_token = self.llm_tokenizer.convert_tokens_to_ids(['assistant'])[0]
        eot_id = self.llm_tokenizer.convert_tokens_to_ids(['<|eot_id|>'])[0]
        
        for i in range(len(input_ids)-1):  # -1 to safely check next token
            if input_ids[i] == start_header_id and input_ids[i+1] == assistant_token:
                # Find next eot_id
                for j in range(i, len(input_ids)):
                    if input_ids[j] == eot_id:
                        assistant_ranges.append({'start': i, 'end': j})
                        break
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'assistant_ranges': assistant_ranges
        }



class SecondStageECGChatDataset(BaseECGDataset):
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
            altered_text = instance['text']
            original_report = ecg_path['report']
            
            return self.prepare_second_input(ecg_signal, altered_text, original_report)
        except Exception as e:
            print(e)
            print(f"Skipping invalid data at index {idx}")
            return None

    def prepare_second_input(self, ecg_signal, altered_text, original_report=None):
        if 'vit' in self.args.model:
            encoder_out = self.encoder_prep.prepare_vit_input(ecg_signal, self.args.num_patches)
        elif 'clip' in self.args.model:
            encoder_out = self.encoder_prep.prepare_clip_input(ecg_signal, original_report)
        elif 'siglip' in self.args.model:
            encoder_out = self.encoder_prep.prepare_siglip_input(ecg_signal, original_report)
        elif 'merl' in self.args.model:
            encoder_out = self.encoder_prep.prepare_merl_input(ecg_signal, original_report)
        elif 'stmem' in self.args.model:
            encoder_out = self.encoder_prep.prepare_st_mem_input(ecg_signal)
        elif 'mtae' in self.args.model:
            encoder_out = self.encoder_prep.prepare_st_mem_input(ecg_signal)
        elif 'mlae' in self.args.model:
            encoder_out = self.encoder_prep.prepare_st_mem_input(ecg_signal)
        if self.args.train == 'second' and self.args.inference is None:
            return self.prepare_training_second(encoder_out, altered_text)
        if self.args.inference == 'second' and self.args.train is None:
            return self.prepare_inference_second(encoder_out, altered_text)
    
    def prepare_training_second(self, encoder_out, altered_text):
        if 'llama' in self.args.model:
            conv = get_conv_template('llama-3')
        conv.set_system_message(self.system_prompt)
        if self.args.data not in [f'ecg_instruct_45k_mapped_{self.args.seg_len}', 
                                  f'ecg_instruct_pulse_mapped_{self.args.seg_len}',
                                  f'ecg_bench_pulse_mapped_{self.args.seg_len}']:
            question, answer = self.get_qa(altered_text)
            altered_text = [{'from': 'human', 'value': question}, {'from': 'assistant', 'value': answer}]
        
        count = 0
        for message in altered_text:
            is_human = message['from'].lower() in ['human', 'user']
            role = conv.roles[0] if is_human else conv.roles[1]
            message_value = message['value'].replace('<ecg>\n', '')
            message_value = message_value.replace('<image>\n', '')
            message_value = message_value.replace('image', 'signal').replace('Image', 'Signal')
            if is_human and count == 0:
                message_value = f"<signal>\n{message_value}"
                count += 1
            conv.append_message(role, message_value)
            
        prompt = conv.get_prompt()
        ecg_position = prompt.find(self.ecg_placeholder)
        prompt_before_ecg = prompt[:ecg_position]
        prompt_after_ecg = prompt[ecg_position + len(self.ecg_placeholder):]
        tokens_before = self.llm_tokenizer.encode(prompt_before_ecg, add_special_tokens=False)
        tokens_after = self.llm_tokenizer.encode(prompt_after_ecg, add_special_tokens=False)
        input_ids = tokens_before + self.signal_id + tokens_after
        labels = [-100] * len(input_ids)
        
        for i, message in enumerate(altered_text):
            if message['from'].lower() == 'assistant':
                response = message['value']
                response_tokens = self.llm_tokenizer.encode(response, add_special_tokens=False)
                for j in range(len(input_ids) - len(response_tokens) + 1):
                    if input_ids[j:j+len(response_tokens)] == response_tokens:
                        labels[j:j+len(response_tokens)] = response_tokens
        eot_id = self.llm_tokenizer.convert_tokens_to_ids('<|eot_id|>')
        for i, token_id in enumerate(input_ids):
            if token_id == eot_id:
                labels[i] = eot_id
        
        input_ids = self.pad_to_max_chat(input_ids)
        signal_id_index = input_ids.index(self.signal_id[0])
        labels = self.pad_to_max_chat(labels)
        labels[labels == self.pad_id] = -100
            
        assert len(input_ids) == self.args.pad_to_max, f"Expected length {self.args.pad_to_max}, got {len(input_ids)}"
        assert len(input_ids) == len(labels), "Tokens and labels length mismatch"
        
        labels = torch.tensor(labels, dtype=torch.int64)    
        position_ids = self.create_position_ids(input_ids)
        attention_mask = self.create_attention_mask(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'labels': labels,
            'position_ids': position_ids,
            'encoder_out': encoder_out,
            'signal_id_index': signal_id_index
        }
    
    def prepare_inference_second(self, encoder_out, altered_text):
        if 'llama' in self.args.model:
            conv = get_conv_template('llama-3')
        conv.set_system_message(self.system_prompt)
        
        if self.args.data not in [f'ecg_instruct_45k_mapped_{self.args.seg_len}', 
                                  f'ecg_instruct_pulse_mapped_{self.args.seg_len}',
                                  f'ecg_bench_pulse_mapped_{self.args.seg_len}']:
            question, answer = self.get_qa(altered_text)
            altered_text = [{'from': 'human', 'value': question}, {'from': 'assistant', 'value': answer}]
        
        
        count = 0
        for message in altered_text:
            is_human = message['from'].lower() in ['human', 'user']
            role = conv.roles[0] if is_human else conv.roles[1]
            message_value = message['value'].replace('<ecg>\n', '')
            message_value = message_value.replace('<image>\n', '')
            message_value = message_value.replace('image', 'signal').replace('Image', 'Signal')
            if is_human and count == 0:
                message_value = f"<signal>\n{message_value}"
                count += 1
            conv.append_message(role, message_value)
            
        prompt = conv.get_prompt()
        ecg_position = prompt.find(self.ecg_placeholder)
        prompt_before_ecg = prompt[:ecg_position]
        prompt_after_ecg = prompt[ecg_position + len(self.ecg_placeholder):]
        tokens_before = self.llm_tokenizer.encode(prompt_before_ecg, add_special_tokens=False)
        tokens_after = self.llm_tokenizer.encode(prompt_after_ecg, add_special_tokens=False)
        input_ids = tokens_before + self.signal_id + tokens_after
        attention_mask = self.create_attention_mask(input_ids)
        
        assistant_ranges = []
        start_header_id = self.llm_tokenizer.convert_tokens_to_ids(['<|start_header_id|>'])[0]
        assistant_token = self.llm_tokenizer.convert_tokens_to_ids(['assistant'])[0]
        eot_id = self.llm_tokenizer.convert_tokens_to_ids(['<|eot_id|>'])[0]
        
        for i in range(len(input_ids)-1):
            if input_ids[i] == start_header_id and input_ids[i+1] == assistant_token:
                for j in range(i, len(input_ids)):
                    if input_ids[j] == eot_id:
                        assistant_ranges.append({'start': i, 'end': j})
                        break
        signal_id_index = input_ids.index(self.signal_id[0])
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'assistant_ranges': assistant_ranges,
            'encoder_out': encoder_out,
            'signal_id_index': signal_id_index
        }
