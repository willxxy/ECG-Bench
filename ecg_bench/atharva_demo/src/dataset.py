
from torch.utils.data import Dataset
from .parse_json import *



class ECGChatDataset(Dataset):
    def __init__(self, json_path, tokenizer = None, args = None, test_mode = False):
        self.json_path = json_path

        with open(self.json_path, 'r') as file:
            self.json_path_list = json.load(file)
        self.tokenizer = tokenizer
        self.test_mode = test_mode
        
    def __len__(self):
        return len(self.json_path_list)
    
    def _parse_conversation_dict_to_list(self, cur_conversation):
        # print("Length of current conversation:",len(cur_conversation["conversations"]))
        agents = {"h":"Human", "g":"Agent"}

        conversation_list = []
        for message in cur_conversation["conversations"]:
            if not self.test_mode and not agents.get(message["from"][0]) == "g":
                # conversation_list.append("[BOM]" + agents.get(message["from"][0],"Unknown") + ": " + message["value"] + "[EOM]")
                conversation_list.append(agents.get(message["from"][0],"Unknown") + ": " + message["value"])

        # Add ECG Signal to return object
        ecg_signal = parse_ecg_signal(cur_conversation["ecg"])
        # print('ecg_signal: ', ecg_signal)
        # print('conversation_list: ', conversation_list)
        conversation_list[0] = conversation_list[0].replace("<ecg>",ecg_signal)
        # print('conversation_list: ', conversation_list)

        return conversation_list

    def __getitem__(self, index):
        try:
            conversation = self.json_path_list[index]
            print('conversation: ', conversation)
        except (FileNotFoundError, ValueError, OSError, KeyError) as e:
            print(f"Error loading files at index {index}: {e}")
            return None

        if conversation is None:
            print(f"Invalid data at index {index}")
            return None


        # print("Batch: ", self._parse_conversation_dict_to_list(conversation))
        "The output here is correct but looks like pytorch's dataloader converts it to individual list elements"
        return self._parse_conversation_dict_to_list(conversation)


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