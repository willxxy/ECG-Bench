from tqdm import tqdm
import torch
import wandb
import torch.distributed as dist
import gc
import random

def trainer(model, tokenizer, dataloader, optimizer, args, epoch, directory_path, instruction = ""):
    model.train()
    num_samples = 0
    dev_count = 0
    total_epoch_loss = 0
    
    count = 0

    for step, batch in enumerate(tqdm(dataloader, desc=f'Training {args.model}', position=0, leave=True)):

        if batch is None:
            print(f"Skipping invalid batch at step {step}")
            continue
        # print(batch)
        optimizer.zero_grad()
        

        chat_history = []
        batch_loss = 0
        print('---' * 50, 'INSTANCE', count, '---' * 50)
        for i in range(0, len(batch) - 1, 2):
            # conversation_list.append("[BOM]" + agents.get(message["from"][0],"Unknown") + ": " + message["value"] + "[EOM]")
            # print(message,type(message))
            instruction = "You are a doctor can you read the signal and answer the question asked by the patient ? "
            # Hugging Face Roberta has max length of 512
            # print(batch)
            # print(i,"out of ", len(batch) - 1)
            question = "[BOM]" + instruction + " ".join(chat_history)[-500:] + batch[i][0] + "[EOM]Agent: "
            answer = "[BOM]" + batch[i+1][0] + "[EOM]"
            chat_history.append(question)

            # print("Check:",message[0][0])
            print('question: ', question)
            print('answer: ', answer)
            
            tokenized_question = tokenizer([question], return_tensors='np', add_special_tokens=False).input_ids[0].tolist()
            attention_mask = tokenizer([question], return_tensors='np', add_special_tokens=False).attention_mask[0].tolist()
            # tokenized_question = tokenizer([question], return_tensors='np', add_special_tokens=False)
            
            tokenized_answer = tokenizer([answer], return_tensors='np', add_special_tokens=False).input_ids[0].tolist()

            # print("Shapes:", len(tokenized_question), len(attention_mask), len(tokenized_answer))
            # print(tokenizer.pad_token_id)
            if len(tokenized_question) > len(tokenized_answer):
                tokenized_answer = tokenized_answer + [tokenizer.pad_token_id]*(len(tokenized_question) - len(tokenized_answer))
            else:
                tokenized_question = [tokenizer.pad_token_id]*(len(tokenized_answer) - len(tokenized_question)) + tokenized_question
                attention_mask = [0]*(len(tokenized_answer) - len(attention_mask)) + attention_mask

            # Hugging Face Roberta has max length of 512
            tokenized_question = tokenized_question[:510]
            tokenized_answer = tokenized_answer[:510]
            tokenized_answer = tokenized_answer[:510]

            # tokenized_answer = tokenizer([answer], return_tensors='np', add_special_tokens=False)

            tokenized_question = torch.tensor(tokenized_question, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
            tokenized_answer = torch.tensor(tokenized_answer, dtype=torch.long).unsqueeze(0)

            # print("Shapes:", tokenized_question.shape, attention_mask.shape, tokenized_answer.shape)

            cur_tokenized_batch = {
                "tokenized_question" : tokenized_question,
                "attention_mask" : attention_mask,
                "tokenized_answer" : tokenized_answer,
            }

            try:
                # print("Current Question:", question)
                optimizer.zero_grad()
                out = model(cur_tokenized_batch)

                logits = out.logits
                generated_ids = torch.argmax(logits, dim=-1)
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                loss = out.loss
                loss.backward()
                # print("Generated : ",generated_text, "\nLoss: ", loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_epoch_loss += loss.item()
                num_samples += 1
                
                random_float = random.random()
                if random_float > 0.9:
                    chat_history.append(answer)
                else:
                    chat_history.append(generated_text)
                # if ((step + 1) % 5000 == 0):
                #     train_model_state_dict = model.state_dict()
                #     train_checkpoint = {
                #         'model': train_model_state_dict,
                #         'epoch': epoch
                #     }
                #     torch.save(train_checkpoint, f'{directory_path}/best_train_model_{epoch}_{step}.pth')
                #     print(f"Best model saved at epoch: {epoch+1} {step}")
                input()
            except Exception as e:
                print(f"Error during training at step {step}: {e}")
                continue
        print('---'*50, 'INSTANCE END ', count, '---'*50)
        count += 1
        
        # if step == 5:
        #     break

    if num_samples == 0:  # Handle case where all batches are invalid
        # print("No valid batches for training.")
        # average_loss = float('inf')
        average_loss = -1
    else:
        average_loss = total_epoch_loss / num_samples
    
    return_dic = {
        'average_loss': average_loss
    }
    # print(return_dic)
    # quit()
    return return_dic

def validater(model, dataloader, args, epoch):
    model.eval()
    total_loss = 0
    num_samples = 0
    dev_count = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc=f'Validating {args.model}', position=0, leave=True)):
            # Skip invalid batches
            if batch is None:
                print(f"Skipping invalid batch at step {step}")
                continue
            
            try:
                out = model(batch)
                loss = out.loss
                total_loss += loss.item()
                num_samples += 1
                
                
                if args.dev:
                    dev_count += 1
                    if dev_count == 10:
                        break
            except Exception as e:
                print(f"Error during validation at step {step}: {e}")
                continue

    if num_samples == 0:  # Handle case where all batches are invalid
        print("No valid batches for validation.")
        average_loss = float('inf')
    else:
        average_loss = total_loss / num_samples
    
    return_dic = {
        'average_loss': average_loss
    }
    return return_dic


import torch
from tqdm import tqdm
import json

# from ecg_byte.utils.model_utils import evaluate_strings

def tester(model, dataloader, tokenizer, args):
    model.eval()
    num_samples = 0
    dev_count = 0
    all_results = []
    gt_answers = []
    gen_answers = []
    questions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc = f'Testing {args.model}', position=0, leave=True):
            if batch is None:
                print(f"Skipping invalid batch ")
                continue
            answer = batch['answer']
            
            try:
                if args.model in ['clip_vit_model', 'clip_model', 'vit_model', 'resnet_model']:
                    out = [model.generate(batch, tokenizer).split('?')[-1]]
                else:
                    out = [model.generate(batch, tokenizer)]
                all_results.append(evaluate_strings(answer, out, args.device))
                gt_answers.append(answer[0])
                gen_answers.append(out[0])
                questions.append(batch['question'][0])
            except Exception as e:
                print('could not evaluate for some reason:', str(e))  # prints the error message
                print(f"Error type: {type(e).__name__}")
                print(out)
                print(answer)
                all_results.append({'BLEU': 0, 
                                    'METEOR': 0.0, 
                                    'ROUGE': {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}, 
                                    'BERTSCORE': {'hf-prec': [0.0], 'hf-rec': [0.0], 'hf-f1': [0.0]}})
            
            num_samples += 1
            
            if args.dev:
                dev_count += 1
                if dev_count == 10:
                    break
                
    # Calculate metrics for this seed
    metric_sums = {
        'BLEU': 0, 'METEOR': 0, 
        'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0,
        'hf-prec': 0, 'hf-rec': 0, 'hf-f1': 0
    }
    metric_counts = {k: 0 for k in metric_sums.keys()}

    for entry in all_results:
        for key, value in entry.items():
            if key in ['ROUGE', 'ROUGE-HF', 'BERTSCORE']:
                for sub_key, sub_value in value.items():
                    if key == 'BERTSCORE':
                        metric_sums[sub_key] += sub_value[0]
                    else:
                        metric_sums[sub_key] += sub_value
                    metric_counts[sub_key] += 1
            else:
                metric_sums[key] += value
                metric_counts[key] += 1

    seed_averages = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
    
    return {
        'metrics': seed_averages,
        'qa_results': {
            'questions': questions,
            'gt_answers': gt_answers,
            'gen_answers': gen_answers
        }
    }
    

'''

import numpy as np
from torch.utils.data import Dataset
import torch
import json
from PIL import Image

from ecg_byte.utils.tokenizer_utils import normalize_all, encode_text
from ecg_byte.utils.file_utils import open_json


def pad_to_max(tokenized_sequence, pad_id, bos_id, eos_id, args):
    if len(tokenized_sequence) > args.pad_to_max:
        truncated_token = tokenized_sequence[:args.pad_to_max]
        full_token = [bos_id] + list(truncated_token) + [eos_id]
        return full_token
    elif len(tokenized_sequence) < args.pad_to_max:
        return [pad_id] * (args.pad_to_max - len(tokenized_sequence)) + [bos_id] + list(tokenized_sequence) + [eos_id]
    else:
        return [bos_id] + list(tokenized_sequence[:args.pad_to_max]) + [eos_id]


def create_attention_like_mask(pad_id, numbers):
    return [0 if num == pad_id else 1 for num in numbers]


def create_position_ids(padded_sequence, pad_token_id):
    padded_sequence = torch.tensor(padded_sequence)
    mask = (padded_sequence != pad_token_id).long()
    position_ids = torch.cumsum(mask, dim=0) - 1
    position_ids.masked_fill_(mask == 0, 0)
    return position_ids
    

class ECGTokenDataset(Dataset):
    def __init__(self, signal_path_list, text_path_list, vocab, merges, tokenizer = None, args = None):
        self.signal_path_list = np.array(signal_path_list)
        self.text_path_list = np.array(text_path_list)
        self.args = args
        self.vocab = vocab
        self.merges = merges
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.bos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        self.eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.sig_start_id = self.tokenizer.convert_tokens_to_ids(['<sig_start>'])
        self.sig_end_id = self.tokenizer.convert_tokens_to_ids(['<sig_end>'])
        self.percentiles = np.load(self.args.percentiles, allow_pickle=True).item()
        
    def __len__(self):
        return len(self.signal_path_list)

    def __getitem__(self, index):
        try:
            signal = np.load(self.signal_path_list[index])
            text_label = open_json(self.text_path_list[index])
        except (FileNotFoundError, ValueError, OSError, KeyError) as e:
            print(f"Error loading files at index {index}: {e}")
            return None

        if signal is None or text_label is None:
            print(f"Invalid data at index {index}")
            return None

        try:
            if self.args.dataset == 'ptb_500':
                question = 'Could you please help me explain my ECG?'
                answer = text_label
            elif self.args.dataset == 'mimic_500':
                question, answer = text_label[0]['value'].replace('\n', '').replace('<ecg>', ''), text_label[1]['value']
            elif self.args.dataset in ['ecg_qa_ptb_500', 'ecg_qa_mimic_500', 'ecg_qa_ptb_250', 'ecg_qa_ptb_1250', 'ecg_qa_ptb_2000']:
                question_type, question, answer = text_label[0], text_label[1], text_label[2]
                answer = ' '.join(answer) if isinstance(answer, list) else answer

            _, normalized_signal = normalize_all(signal, percentiles=self.percentiles)
            string_signal = ''.join(normalized_signal.flatten())
            tokenized_signal = encode_text(string_signal, self.merges)

            tokenized_question = self.tokenizer([question], return_tensors='np', add_special_tokens=False).input_ids[0].tolist()
            tokenized_answer = self.tokenizer([answer], return_tensors='np', add_special_tokens=False).input_ids[0].tolist()
            tokenized_signal = self.tokenizer.convert_tokens_to_ids([f'signal_{ids}' for ids in tokenized_signal])
        except Exception as e:
            print(f"Error processing data at index {index}: {e}")
            return None

        if self.args.inference:
            return self._prepare_inference(tokenized_signal, tokenized_question, answer, question)
        else:
            return self._prepare_training(tokenized_signal, tokenized_question, tokenized_answer, signal, normalized_signal, string_signal)


    def _prepare_inference(self, tokenized_signal, tokenized_question, answer, question):
        inference_seq = [self.bos_id] + self.sig_start_id + tokenized_signal + self.sig_end_id + tokenized_question
        attention_mask = create_attention_like_mask(self.pad_id, inference_seq)
        return {
            'answer': answer,
            'question': question,
            'tokenized_signal': torch.tensor(inference_seq, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32)
        }

    def _prepare_training(self, tokenized_signal, tokenized_question, tokenized_answer, signal, normalized_signal, string_signal):

        qa_len = len(tokenized_question) + len(tokenized_answer)
        available_space = self.args.pad_to_max - qa_len

        if len(tokenized_signal) > available_space:
            tokenized_signal = [self.bos_id] + self.sig_start_id + tokenized_signal[:available_space] + self.sig_end_id
        elif len(tokenized_signal) < available_space:
            tokenized_signal = [self.pad_id] * (available_space - len(tokenized_signal)) + [self.bos_id] + self.sig_start_id + tokenized_signal + self.sig_end_id
        else:
            tokenized_signal = [self.bos_id] + self.sig_start_id + tokenized_signal + self.sig_end_id

        full_seq = tokenized_signal + tokenized_question + tokenized_answer
        padded_masked_sample = full_seq + [self.eos_id]

        padded_quantized_signal_ids_input = [-100]* (len(tokenized_signal) + len(tokenized_question)) + tokenized_answer + [self.eos_id]

        padded_quantized_signal_ids_input = torch.tensor(padded_quantized_signal_ids_input, dtype=torch.int64)

        position_ids = create_position_ids(padded_masked_sample, self.pad_id)
        attention_mask = create_attention_like_mask(self.pad_id, padded_masked_sample)

        assert len(padded_masked_sample) == len(attention_mask) == (self.args.pad_to_max + 4), \
            f"Lengths don't match: masked_sample ({len(padded_masked_sample)}), attention_mask ({len(attention_mask)})"

        return {
            'tokenized_signal': torch.tensor(padded_masked_sample, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'quantized_signal_ids_input': padded_quantized_signal_ids_input,
            'position_ids': position_ids,
            'signal': signal,
        }
    

    
class ECGCLIPPretrain(Dataset):
    def __init__(self, signal_path_list, text_path_list, clip_tokenizer = None, vit_tokenizer = None, resnet_tokenizer = None, args = None):
        self.signal_path_list = np.array(signal_path_list)
        self.text_path_list = np.array(text_path_list)
        self.clip_tokenizer = clip_tokenizer
        self.vit_tokenizer = vit_tokenizer
        self.resnet_tokenizer = resnet_tokenizer
        self.args = args
        
    def __len__(self):
        return len(self.signal_path_list)

    def __getitem__(self, index):
        signal = np.load(self.signal_path_list[index])
        text_label = json.load(open(self.text_path_list[index]))[1]['value']
        
        signal_min, signal_max = signal.min(), signal.max()
        norm_signal = ((signal - signal_min) / (signal_max - signal_min + 1e-6))
        norm_signal *= 1000
        normalized_signal = ((signal - signal_min) / (signal_max - signal_min + 1e-6)) * 255
        image_signal = np.stack([normalized_signal] * 3, axis=-1).astype(np.uint8)
        image_signal = Image.fromarray(image_signal)
        
        ### placeholder 
        mask = 1
        vit_inputs = 1
        clip_input_ids = 1
        clip_attention_mask = 1
        vit_pixel_values = 1
        clip_pixel_values = 1
        resnet_input_ids = 1
        resnet_attention_mask =1
        
        if self.args.model == 'clip':
            inputs = self.clip_tokenizer(text = [text_label], images = [image_signal], return_tensors = 'pt', padding = 'max_length', max_length=77, truncation=True)
            clip_input_ids = inputs['input_ids'][0]
            clip_attention_mask = inputs['attention_mask'][0]
            clip_pixel_values = inputs['pixel_values'][0]
        elif self.args.model == 'vit':
            mask = torch.rand(size=(1, self.args.num_patches)) < 0.75
            mask = mask.squeeze(0)
            inputs = self.vit_tokenizer(images=image_signal, return_tensors="pt")
            vit_pixel_values = inputs['pixel_values'][0]
        elif self.args.model == 'clip_vit':
            mask = torch.rand(size=(1, self.args.num_patches)) < 0.75
            mask = mask.squeeze(0)
            vit_inputs = self.vit_tokenizer(images=image_signal, return_tensors="pt")
            clip_inputs = self.clip_tokenizer(text = [text_label], images = [image_signal], return_tensors = 'pt', padding = 'max_length', max_length=77, truncation=True)
            clip_input_ids = clip_inputs['input_ids'][0]
            clip_attention_mask = clip_inputs['attention_mask'][0]
            vit_pixel_values = vit_inputs['pixel_values'][0]
            clip_pixel_values = clip_inputs['pixel_values'][0]
        elif self.args.model == 'resnet':
            resnet_inputs = self.resnet_tokenizer(text = [text_label], return_tensors = 'pt', padding = 'max_length', max_length=64, truncation=True)
            resnet_input_ids = resnet_inputs['input_ids'][0]
            resnet_attention_mask = resnet_inputs['attention_mask'][0]

        return_dic = {
            'clip_input_ids': clip_input_ids,
            'clip_att_mask': clip_attention_mask,
            'vit_pixel': vit_pixel_values,
            'clip_pixel': clip_pixel_values,
            'mask': mask,
            'norm_signal': norm_signal.astype(np.float32),
            'resnet_input_ids' : resnet_input_ids,
            'resnet_att_mask' : resnet_attention_mask
        }
        
        return return_dic
    
    

class ECGCLIPFinetune(Dataset):
    def __init__(self, signal_path_list, text_path_list, tokenizer = None, 
                 processor_clip = None, processor_vit = None, args = None):
        self.signal_path_list = np.array(signal_path_list)
        self.text_path_list = np.array(text_path_list)
        self.args = args
        self.tokenizer = tokenizer
        self.processor_clip = processor_clip
        self.processor_vit = processor_vit
        self.pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.bos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        self.eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.sig_start_id = self.tokenizer.convert_tokens_to_ids(['<sig_start>'])
        self.sig_end_id = self.tokenizer.convert_tokens_to_ids(['<sig_end>'])
        self.signal_id = self.tokenizer.convert_tokens_to_ids(['<signal>'])
        
    def __len__(self):
        return len(self.signal_path_list)

    def __getitem__(self, index):
        signal = np.load(self.signal_path_list[index])
        text_label = open_json(self.text_path_list[index])
        # print(text_label)
        if self.args.dataset == 'ptb_500':
            question = 'Could you please help me explain my ECG?'
            answer = text_label
        elif self.args.dataset == 'mimic_500':
            question, answer = text_label[0]['value'].replace('\n', '').replace('<ecg>', ''), text_label[1]['value']
        elif self.args.dataset in ['ecg_qa_ptb_500', 'ecg_qa_mimic_500', 'ecg_qa_ptb_250', 'ecg_qa_ptb_1250', 'ecg_qa_ptb_2000']:
            question_type, question, answer = text_label[0], text_label[1], text_label[2]
            answer = ' '.join(answer) if isinstance(answer, list) else answer
        
        signal_min, signal_max = signal.min(), signal.max()
        norm_signal = ((signal - signal_min) / (signal_max - signal_min + 1e-6))
        norm_signal *= 1000 # how MERL scales
        normalized_signal = ((signal - signal_min) / (signal_max - signal_min + 1e-6)) * 255
        image_signal = np.stack([normalized_signal] * 3, axis=-1).astype(np.uint8)
        image_signal = Image.fromarray(image_signal)
        
        ### Placeholder
        mask = 1
        clip_pixel = 1
        clip_att_mask = 1
        vit_pixel = 1
        clip_input_ids = 1
        
        if self.args.model == 'clip_model':
            inputs = self.processor_clip(text = [answer], images = [image_signal], return_tensors = 'pt', padding = 'max_length', max_length=77, truncation=True)
            clip_input_ids = inputs['input_ids'][0]
            clip_att_mask = inputs['attention_mask'][0]
            clip_pixel = inputs['pixel_values'][0]
        elif self.args.model == 'vit_model':
            mask = torch.rand(size=(1, self.args.num_patches)) < 0.75
            mask = mask.squeeze(0)
            inputs = self.processor_vit(images=image_signal, return_tensors="pt")
            vit_pixel = inputs['pixel_values'][0]
        elif self.args.model == 'clip_vit_model':
            clip_inputs = self.processor_clip(text = [answer], images = [image_signal], return_tensors = 'pt', padding = 'max_length', max_length=77, truncation=True)
            clip_input_ids = clip_inputs['input_ids'][0]
            clip_att_mask = clip_inputs['attention_mask'][0]
            clip_pixel = clip_inputs['pixel_values'][0]
            vit_inputs = self.processor_vit(images=image_signal, return_tensors="pt")
            vit_pixel = vit_inputs['pixel_values'][0]
            mask = torch.rand(size=(1, self.args.num_patches)) < 0.75
            mask = mask.squeeze(0)
        
        tokenized_question = self.tokenizer([question], return_tensors='np', add_special_tokens=False).input_ids[0].tolist()
        tokenized_answer = self.tokenizer([answer], return_tensors='np', add_special_tokens=False).input_ids[0].tolist()
                
        if self.args.inference:
            return self._prepare_inference(tokenized_question, answer, question,
                                           clip_input_ids, clip_att_mask, clip_pixel, vit_pixel, mask, norm_signal)
        else:
            return self._prepare_training(tokenized_question, tokenized_answer, 
                                          clip_input_ids, clip_att_mask, clip_pixel, vit_pixel, mask, norm_signal)

    def _prepare_inference(self, tokenized_question, answer, question, 
                           clip_input_ids = None, clip_att_mask = None, clip_pixel = None, vit_pixel = None, mask = None,
                           norm_signal = None):
        inference_seq = [self.bos_id] + self.sig_start_id + self.sig_end_id + tokenized_question
        inference_seq2= [self.bos_id] + self.sig_start_id + self.signal_id + self.sig_end_id + tokenized_question
        
        attention_mask = create_attention_like_mask(self.pad_id, inference_seq)
        return {
            'answer': answer,
            'question': question,
            'tokenized_signal': torch.tensor(inference_seq, dtype=torch.int64),
            'tokenized_signal2': torch.tensor(inference_seq2, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'clip_input_ids' : clip_input_ids,
            'clip_att_mask' : clip_att_mask,
            'clip_pixel' : clip_pixel,
            'vit_pixel' : vit_pixel,
            'mask' : mask,
            'norm_signal': norm_signal.astype(np.float32),
        }

    def _prepare_training(self, tokenized_question, tokenized_answer, 
                          clip_input_ids = None, clip_att_mask = None, clip_pixel = None, vit_pixel = None, mask = None,
                          norm_signal = None):
        full_seq = self.sig_start_id + self.signal_id + self.sig_end_id + tokenized_question + tokenized_answer        
        labels = [-100] * (len(tokenized_question) + 3) + tokenized_answer
        
        padded_masked_sample = pad_to_max(full_seq, self.pad_id, self.bos_id, self.eos_id, self.args)
        position_ids = create_position_ids(padded_masked_sample, self.pad_id)
        padded_quantized_signal_ids_input = pad_to_max(labels, self.pad_id, self.bos_id, self.eos_id, self.args)
        padded_quantized_signal_ids_input[padded_quantized_signal_ids_input == self.pad_id] = -100
        padded_quantized_signal_ids_input[padded_quantized_signal_ids_input == self.bos_id] = -100
        attention_mask = create_attention_like_mask(self.pad_id, padded_masked_sample)
        
        assert len(padded_masked_sample) == len(attention_mask) == (self.args.pad_to_max + 2), \
            f"Lengths don't match: masked_sample ({len(padded_masked_sample)}), attention_mask ({len(attention_mask)}) {self.args.pad_to_max + 2}"
        
        return {
            'tokenized_signal': torch.tensor(padded_masked_sample, dtype=torch.int64),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'quantized_signal_ids_input': torch.tensor(padded_quantized_signal_ids_input, dtype=torch.int64),
            'position_ids': position_ids,
            'clip_input_ids' : clip_input_ids,
            'clip_att_mask' : clip_att_mask,
            'clip_pixel' : clip_pixel,
            'vit_pixel' : vit_pixel,
            'mask' : mask,
            'norm_signal': norm_signal.astype(np.float32),
        }
    
'''