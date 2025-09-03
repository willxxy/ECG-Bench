import random

import numpy as np
import torch
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset
from typing import Sequence, List, Literal

from ecg_bench.utils.conversation_utils import get_conv_template


class BaseECGDataset(Dataset):
    def __init__(self, json_data_file, train_utils, encoder_tokenizer=None, llm_tokenizer=None):
        self.json_data_file = json_data_file
        self.train_utils = train_utils
        self.viz = train_utils.viz
        self.args = self.train_utils.args
        self.encoder_tokenizer = encoder_tokenizer
        self.llm_tokenizer = llm_tokenizer
        if self.args.rag:
            from ecg_bench.utils.rag_utils import RAGECGDatabase
            self.args.create_rag_db = None
            self.rag_db = RAGECGDatabase(self.args, self.train_utils.fm)
        if llm_tokenizer is not None:
            self.pad_id = self.llm_tokenizer.convert_tokens_to_ids(self.llm_tokenizer.pad_token)
            self.signal_id = self.llm_tokenizer.convert_tokens_to_ids(["<signal>"])
        if self.args.train == "end2end" or self.args.inference == "end2end" or self.args.train == "second" or self.args.inference == "second":
            self.system_prompt = self.train_utils.fm.get_system_prompt(self.args.system_prompt)
            self.ecg_placeholder = "<signal>"

    def __len__(self): return len(self.json_data_file)

    def signal_to_image(self, signal):
        if self.args.image:
            image = self.viz.get_plot_as_image(signal, self.args.target_sf)
            if self.args.augment_image and random.random() < 0.6:
                return self.augment_image(image)
            return Image.fromarray(image)
        if self.args.instance_normalize:
            normalized_signal, _, _ = self.train_utils.ecg_tokenizer_utils.instance_normalize(signal)
        rgb_norm_signal = np.stack([normalized_signal * 255] * 3, axis=-1).astype(np.uint8)
        return Image.fromarray(rgb_norm_signal)

    def perturb_signal(self, signal):
        if random.random() < 0.5:  # 50% chance of perturbation
            noise_level = 0.05
            noise = np.random.normal(0, noise_level * np.std(signal), signal.shape)
            perturbed_signal = signal + noise

            if random.random() < 0.5:
                wander_amplitude = 0.07 * np.max(np.abs(signal))
                wander = wander_amplitude * np.sin(np.linspace(0, random.randint(1, 5) * np.pi, signal.shape[1]))
                wander = np.tile(wander, (signal.shape[0], 1))
                perturbed_signal += wander

            return perturbed_signal
        return signal

    def blackout_signal(self, signal):
        return np.zeros_like(signal)

    def augment_image(self, image):
        seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),    # 50% chance to change brightness
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-5, 5))),    # 50% chance to rotate by -5° to 5°
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 1.5))),  # 50% chance to apply Gaussian blur
        iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-30, 30), per_channel=True)),  # 50% chance to adjust hue
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
        if self.args.data == f"pretrain-mimic-{self.args.target_sf}-{self.args.seg_len}":
            question, answer = altered_text[0]["value"].replace("\n", "").replace("<ecg>", ""), altered_text[1]["value"]
        elif self.args.data in [f"ecg-qa-mimic-iv-ecg-{self.args.target_sf}-{self.args.seg_len}", f"ecg-qa-ptbxl-{self.args.target_sf}-{self.args.seg_len}"]:
            question_type, question, answer = altered_text[0], altered_text[1], altered_text[2]
            answer = " ".join(answer) if isinstance(answer, list) else answer
        return question, answer

    def pad_to_max_chat(
        self,
        tokens: Sequence[int],
        pad_side: Literal["left", "right"] = "left",
    ) -> List[int]:
        """Pad to args.pad_to_max with self.pad_id. Truncates to first max_len tokens.
        pad_side: 'left' (default) or 'right' controls where padding is added.
        """
        max_len = self.args.pad_to_max
        seq = list(tokens)[:max_len]                 # truncate if longer
        pad_len = max_len - len(seq)
        if pad_len <= 0: return seq

        pad = [self.pad_id] * pad_len
        if pad_side not in ("left", "right"):
            raise ValueError("pad_side must be 'left' or 'right'.")
        return pad + seq if pad_side == "left" else seq + pad

    def setup_conversation_template(self, signal = None):
        if "llama" in self.args.model:
            conv = get_conv_template("llama-3")
        elif "qwen" in self.args.model:
            conv = get_conv_template("qwen-7b-chat")
        elif "gemma" in self.args.model:
            conv = get_conv_template("gemma")

        if "gemma" not in self.args.model and ("qwen" in self.args.model or "llama" in self.args.model):
            if self.args.rag:
                rag_results = self.rag_db.search_similar(query_signal=signal, k=self.args.rag_k, mode="signal")
                filtered_rag_results = self.rag_db.format_search(rag_results)
                modified_system_prompt = f"{self.system_prompt}\n{filtered_rag_results}"
                if self.args.dev:
                    print("filtered_rag_results", filtered_rag_results)
                    print("modified_system_prompt", modified_system_prompt)

                conv.set_system_message(modified_system_prompt)
            else:
                conv.set_system_message(self.system_prompt)
        return conv

    def process_altered_text(self, altered_text):
        if self.args.data not in [f"ecg-instruct-45k-{self.args.target_sf}-{self.args.seg_len}",
                                  f"ecg-instruct-pulse-{self.args.target_sf}-{self.args.seg_len}",
                                  f"ecg-bench-pulse-{self.args.target_sf}-{self.args.seg_len}"]:
            question, answer = self.get_qa(altered_text)
            if "gemma" in self.args.model:
                altered_text = [{"from": "human", "value": question}, {"from": "model", "value": answer}]
            else:
                altered_text = [{"from": "human", "value": question}, {"from": "assistant", "value": answer}]
        return altered_text

    def append_messages_to_conv(self, conv, altered_text, signal=None):
        count = 0
        for message in altered_text:
            is_human = message["from"].lower() in ["human", "user"]
            role = conv.roles[0] if is_human else conv.roles[1]
            message_value = message["value"].replace("<ecg>\n", "")
            message_value = message_value.replace("<image>\n", "")
            message_value = message_value.replace("<image>", "")
            message_value = message_value.replace("<ecg>", "")
            message_value = message_value.replace("image", "signal").replace("Image", "Signal")
            if is_human and count == 0:
                message_value = f"<signal>\n{message_value}"
                count += 1
            conv.append_message(role, message_value)
        return conv

    def get_input_tokens(self, conv):
        prompt = conv.get_prompt()
        if self.args.dev: print("prompt\n", prompt)
        ecg_position = prompt.find(self.ecg_placeholder)
        prompt_before_ecg = prompt[:ecg_position]
        prompt_after_ecg = prompt[ecg_position + len(self.ecg_placeholder):]
        tokens_before = self.llm_tokenizer.encode(prompt_before_ecg, add_special_tokens=False)
        tokens_after = self.llm_tokenizer.encode(prompt_after_ecg, add_special_tokens=False)
        return tokens_before, tokens_after

    def get_special_token_ids(self):
        if "llama" in self.args.model:
            start_header_id = self.llm_tokenizer.convert_tokens_to_ids(["<|start_header_id|>"])[0]
            eot_id = self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        elif "gemma" in self.args.model:
            start_header_id = self.llm_tokenizer.convert_tokens_to_ids(["<start_of_turn>"])[0]
            eot_id = self.llm_tokenizer.convert_tokens_to_ids("<end_of_turn>")
        elif "qwen" in self.args.model:
            start_header_id = self.llm_tokenizer.convert_tokens_to_ids(["<|im_start|>"])[0]
            eot_id = self.llm_tokenizer.convert_tokens_to_ids("<|im_end|>")

        if "gemma" in self.args.model:
            assistant_token = self.llm_tokenizer.convert_tokens_to_ids(["model"])[0]
        else:
            assistant_token = self.llm_tokenizer.convert_tokens_to_ids(["assistant"])[0]

        return start_header_id, assistant_token, eot_id

    def find_assistant_ranges(self, input_ids):
        start_header_id, assistant_token, eot_id = self.get_special_token_ids()
        assistant_ranges = []

        for i in range(len(input_ids)-1):  # -1 to safely check next token
            if input_ids[i] == start_header_id and input_ids[i+1] == assistant_token:
                # Find next eot_id
                for j in range(i, len(input_ids)):
                    if input_ids[j] == eot_id:
                        assistant_ranges.append({"start": i, "end": j})
                        break

        return assistant_ranges

    def create_labels_from_responses(self, input_ids, altered_text):
        labels = [-100] * len(input_ids)
        _, _, eot_id = self.get_special_token_ids()

        assistant_roles = {"assistant", "model", "gpt"}
        responses = []
        for m in altered_text:
            if m["from"].lower() in assistant_roles:
                toks = self.llm_tokenizer.encode(m["value"], add_special_tokens=False)
                if toks:
                    responses.append(toks)
        if not responses:
            for i, t in enumerate(input_ids):
                if t == eot_id:
                    labels[i] = eot_id
            return labels

        start_tokens = {t[0] for t in responses}
        positions = {st: [] for st in start_tokens}
        eot_positions = []
        for i, t in enumerate(input_ids):
            if t in positions:
                positions[t].append(i)
            if t == eot_id:
                eot_positions.append(i)

        n = len(input_ids)
        for toks in responses:
            first = toks[0]
            for s in positions.get(first, []):
                e = s + len(toks)
                if e <= n and input_ids[s:e] == toks:
                    labels[s:e] = toks
                    break

        for i in eot_positions:
            labels[i] = eot_id

        return labels


    def token_to_ids(self, labels):
        labels_np = np.array(labels)
        non_neg_indices = np.where(labels_np != -100)[0]
        if len(non_neg_indices) > 0:
            non_neg_values = labels_np[non_neg_indices].tolist()
            tokens = self.llm_tokenizer.convert_ids_to_tokens(non_neg_values)
            for idx, (token, token_id) in enumerate(zip(tokens, non_neg_values)):
                print(f"{idx}: {token} -> {token_id}")
        else:
            print("No valid labels found (all are -100)")
        print("="*100)


class EncoderInputPreparation(BaseECGDataset):
    def __init__(self, encoder_tokenizer, train_utils):
        super().__init__(json_data_file=None, train_utils=train_utils, encoder_tokenizer=encoder_tokenizer)

    def prepare_signal_input(self, ecg_signal):
        if self.args.instance_normalize:
            normalized_signal, _, _ = self.train_utils.ecg_tokenizer_utils.instance_normalize(ecg_signal)
        return {"signal": normalized_signal.astype(np.float32),
                "orig_signal": ecg_signal.astype(np.float32)}

    def prepare_vit_input(self, ecg_signal, num_patches):
        image_signal = self.signal_to_image(ecg_signal)
        vit_inputs = self.encoder_tokenizer(images=image_signal,
                                          return_tensors="pt")
        pixel_values = vit_inputs["pixel_values"][0].contiguous()
        mask = torch.rand(size=(1, num_patches)) < 0.75
        return {
            "vit_pixel": pixel_values,
            "vit_mask": mask[0].contiguous(),
            "orig_signal": ecg_signal.astype(np.float32),
        }

    def prepare_clip_input(self, ecg_signal, original_report):
        image_signal = self.signal_to_image(ecg_signal)
        clip_inputs = self.encoder_tokenizer(text=[original_report],
                                           images=[image_signal],
                                           return_tensors="pt",
                                           padding="max_length",
                                           max_length=77,
                                           truncation=True)
        return {
            "clip_input_ids": clip_inputs["input_ids"][0].contiguous(),
            "clip_att_mask": clip_inputs["attention_mask"][0].contiguous(),
            "clip_pixel": clip_inputs["pixel_values"][0].contiguous(),
            "orig_signal": ecg_signal.astype(np.float32),
        }

    def prepare_dinov2_input(self, ecg_signal):
        image_signal = self.signal_to_image(ecg_signal)
        dinov2_inputs = self.encoder_tokenizer(images=image_signal,
                                           return_tensors="pt")
        pixel_values = dinov2_inputs["pixel_values"][0].contiguous()
        return {"dinov2_pixel": pixel_values,
                "orig_signal": ecg_signal.astype(np.float32)}

    def prepare_siglip_input(self, ecg_signal, original_report):
        image_signal = self.signal_to_image(ecg_signal)
        siglip_inputs = self.encoder_tokenizer(text=[original_report],
                                           images=[image_signal],
                                           return_tensors="pt",
                                           padding="max_length",
                                           max_length=64,
                                           truncation=True)
        ### siglip does not have attention mask??
        pad_token_id = 1 # so we define the pad token manually (This is the id for pad in siglip)
        attention_mask = (siglip_inputs["input_ids"][0] != pad_token_id).int()
        return {
            "siglip_input_ids": siglip_inputs["input_ids"][0].contiguous(),
            "siglip_att_mask": attention_mask.contiguous(),
            "siglip_pixel": siglip_inputs["pixel_values"][0].contiguous(),
            "orig_signal": ecg_signal.astype(np.float32),
        }

    def prepare_merl_input(self, ecg_signal, original_report):
        if self.args.instance_normalize:
            normalized_signal, _, _ = self.train_utils.ecg_tokenizer_utils.instance_normalize(ecg_signal)
        merl_inputs = self.encoder_tokenizer(text=[original_report],
                                           return_tensors="pt",
                                           padding="max_length",
                                           max_length=64,
                                           truncation=True)
        return {
            "merl_input_ids": merl_inputs["input_ids"][0].contiguous(),
            "merl_att_mask": merl_inputs["attention_mask"][0].contiguous(),
            "signal": normalized_signal.astype(np.float32),
            "orig_signal": ecg_signal.astype(np.float32),
        }


class FirstStageECGDataset(BaseECGDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_prep = EncoderInputPreparation(self.encoder_tokenizer, self.train_utils)

    def __getitem__(self, idx):
        try:
            instance = self.json_data_file[idx]
            np_path = instance["ecg_path"]
            ecg_path = self.train_utils.fm.open_npy(np_path)
            ecg_signal = ecg_path["ecg"]
            original_report = ecg_path["report"]

            if "clip" in self.args.model:
                return self.encoder_prep.prepare_clip_input(ecg_signal, original_report)
            if "vit" in self.args.model:
                return self.encoder_prep.prepare_vit_input(ecg_signal, self.args.num_patches)
            if "siglip" in self.args.model:
                return self.encoder_prep.prepare_siglip_input(ecg_signal, original_report)
            if "merl" in self.args.model:
                return self.encoder_prep.prepare_merl_input(ecg_signal, original_report)
            if self.args.model in ["stmem", "mtae", "mlae"]:
                return self.encoder_prep.prepare_signal_input(ecg_signal)
        except Exception as e:
            print(e)
            print(f"Skipping invalid data at index {idx}")
            return None


class End2EndECGChatDataset(BaseECGDataset):
    def __getitem__(self, idx):
        try:
            instance = self.json_data_file[idx]
            np_path = instance["ecg_path"]
            ecg_path = self.train_utils.fm.open_npy(np_path)
            ecg_signal = ecg_path["ecg"]
            if self.args.perturb:
                ecg_signal = self.perturb_signal(ecg_signal)
            if self.args.blackout:
                ecg_signal = self.blackout_signal(ecg_signal)
            altered_text = instance["text"]
            return self.prepare_end2end_input(ecg_signal, altered_text)
        except Exception as e:
            print(e)
            print(f"Skipping invalid data at index {idx}")
            return None

    def prepare_end2end_input(self, ecg_signal, altered_text):
        if self.args.train == "end2end" and self.args.inference is None:
            return self.prepare_training_end2end(ecg_signal, altered_text)
        if self.args.inference == "end2end" and self.args.train is None:
            return self.prepare_inference_end2end(ecg_signal, altered_text)

    def prepare_training_end2end(self, ecg_signal, altered_text):
        conv = self.setup_conversation_template(signal=ecg_signal)
        altered_text = self.process_altered_text(altered_text)
        conv = self.append_messages_to_conv(conv, altered_text, ecg_signal)

        tokens_before, tokens_after = self.get_input_tokens(conv)

        symbol_signal = self.train_utils.ecg_tokenizer_utils._to_symbol_string(ecg_signal)
        encoded_signal = self.train_utils.ecg_tokenizer_utils.encode_symbol(symbol_signal,
                                                                          self.train_utils.ecg_tokenizer_utils.merges)
        tokenized_signal = self.llm_tokenizer.convert_tokens_to_ids([f"signal_{ids}" for ids in encoded_signal])
        signal_tokens = tokenized_signal[:min(500, len(tokenized_signal))] # keep at least 500, unless tokenized signal is shorter
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

        labels = self.create_labels_from_responses(input_ids, altered_text)

        if self.args.dev:
            self.token_to_ids(labels)

        assert len(input_ids) == self.args.pad_to_max, f"Expected length {self.args.pad_to_max}, got {len(input_ids)}"

        labels = torch.tensor(labels, dtype=torch.int64)
        position_ids = self.create_position_ids(input_ids)
        attention_mask = self.create_attention_mask(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "attn_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "labels": labels,
            "position_ids": position_ids,
            "signal": ecg_signal,
        }

    def prepare_inference_end2end(self, ecg_signal, altered_text):
        conv = self.setup_conversation_template(signal=ecg_signal)
        altered_text = self.process_altered_text(altered_text)
        conv = self.append_messages_to_conv(conv, altered_text, ecg_signal)

        tokens_before, tokens_after = self.get_input_tokens(conv)

        symbol_signal = self.train_utils.ecg_tokenizer_utils._to_symbol_string(ecg_signal)
        encoded_signal = self.train_utils.ecg_tokenizer_utils.encode_symbol(symbol_signal,
                                                                          self.train_utils.ecg_tokenizer_utils.merges)
        tokenized_signal = self.llm_tokenizer.convert_tokens_to_ids([f"signal_{ids}" for ids in encoded_signal])

        input_ids = tokens_before + tokenized_signal + tokens_after
        attention_mask = self.create_attention_mask(input_ids)

        # Find assistant response ranges
        assistant_ranges = self.find_assistant_ranges(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "attn_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "assistant_ranges": assistant_ranges,
        }


class SecondStageECGChatDataset(BaseECGDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_prep = EncoderInputPreparation(self.encoder_tokenizer, self.train_utils)

    def __getitem__(self, idx):
        try:
            instance = self.json_data_file[idx]
            np_path = instance["ecg_path"]
            ecg_path = self.train_utils.fm.open_npy(np_path)
            ecg_signal = ecg_path["ecg"]
            if self.args.perturb:
                ecg_signal = self.perturb_signal(ecg_signal)
            if self.args.blackout:
                ecg_signal = self.blackout_signal(ecg_signal)
            altered_text = instance["text"]
            original_report = ecg_path["report"]

            return self.prepare_second_input(ecg_signal, altered_text, original_report)
        except Exception as e:
            print(e)
            print(f"Skipping invalid data at index {idx}")
            return None

    def prepare_second_input(self, ecg_signal, altered_text, original_report=None):
        if "vit" in self.args.model:
            encoder_out = self.encoder_prep.prepare_vit_input(ecg_signal, self.args.num_patches)
        elif "clip" in self.args.model:
            encoder_out = self.encoder_prep.prepare_clip_input(ecg_signal, original_report)
        elif "siglip" in self.args.model:
            encoder_out = self.encoder_prep.prepare_siglip_input(ecg_signal, original_report)
        elif "merl" in self.args.model:
            encoder_out = self.encoder_prep.prepare_merl_input(ecg_signal, original_report)
        elif any(key in self.args.model for key in ("stmem", "mtae", "mlae", "encoderfree")):
            encoder_out = self.encoder_prep.prepare_signal_input(ecg_signal)

        if self.args.train == "second" and self.args.inference is None:
            return self.prepare_training_second(encoder_out, altered_text)
        if self.args.inference == "second" and self.args.train is None:
            return self.prepare_inference_second(encoder_out, altered_text)

    def prepare_training_second(self, encoder_out, altered_text):
        conv = self.setup_conversation_template(signal=encoder_out["orig_signal"])
        altered_text = self.process_altered_text(altered_text)
        conv = self.append_messages_to_conv(conv, altered_text, encoder_out["orig_signal"])

        tokens_before, tokens_after = self.get_input_tokens(conv)

        input_ids = tokens_before + self.signal_id + tokens_after
        labels = self.create_labels_from_responses(input_ids, altered_text)

        input_ids = self.pad_to_max_chat(input_ids)
        signal_id_index = input_ids.index(self.signal_id[0])
        labels = torch.tensor(self.pad_to_max_chat(labels), dtype=torch.int64)
        labels[labels == self.pad_id] = -100

        assert len(input_ids) == self.args.pad_to_max, f"Expected length {self.args.pad_to_max}, got {len(input_ids)}"
        assert len(input_ids) == len(labels), "Tokens and labels length mismatch"

        if self.args.dev:
            self.token_to_ids(labels)

        position_ids = self.create_position_ids(input_ids)
        attention_mask = self.create_attention_mask(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "attn_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "labels": labels,
            "position_ids": position_ids,
            "encoder_out": encoder_out,
            "signal_id_index": signal_id_index,
        }

    def prepare_inference_second(self, encoder_out, altered_text):
        conv = self.setup_conversation_template(signal=encoder_out["orig_signal"])
        altered_text = self.process_altered_text(altered_text)
        conv = self.append_messages_to_conv(conv, altered_text, encoder_out["orig_signal"])

        tokens_before, tokens_after = self.get_input_tokens(conv)

        input_ids = tokens_before + self.signal_id + tokens_after
        attention_mask = self.create_attention_mask(input_ids)

        assistant_ranges = self.find_assistant_ranges(input_ids)
        signal_id_index = input_ids.index(self.signal_id[0])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "attn_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "assistant_ranges": assistant_ranges,
            "encoder_out": encoder_out,
            "signal_id_index": signal_id_index,
        }
