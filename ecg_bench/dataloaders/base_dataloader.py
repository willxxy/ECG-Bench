from torch.utils.data import Dataset
import random
import numpy as np
from typing import List, Tuple
from PIL import Image
import torch

from ecg_bench.configs.constants import HF_LLMS
from ecg_bench.utils.file_manager import FileManager
from ecg_bench.utils.chat_template import get_conv_template
from ecg_bench.configs.constants import (
    SIGNAL_TOKEN_PLACEHOLDER,
    LEADING_PREFIX_RE,
    TAG_RE,
    IMAGE_WORD_RE,
    case_preserving_signal,
    ECG_ENCODERS,
    VISION_ENCODERS,
)
from ecg_bench.utils.gpu_setup import is_main


class BaseECGDataset(Dataset):
    def __init__(self, data, mode, args):
        self.data = data
        self.args = args
        self.mode = mode
        self.fm = FileManager()
        self.normalize_epsilon = 1e-6
        if self.args.llm:
            self.chat_template = self.make_chat_template()
        if self.args.encoder:
            if self.args.encoder in ECG_ENCODERS:
                self.max_len = ECG_ENCODERS[self.args.encoder]["encoder_input_len"]
            elif self.args.encoder in VISION_ENCODERS:
                self.max_len = VISION_ENCODERS[self.args.encoder]["encoder_input_len"]

    def __len__(self):
        return len(self.data)

    ### ENCODER TRAINING FUNCTIONS ###
    def prepare_clip_input(self, diagnostic_report: str, ecg_image: Image.Image, ecg_signal: np.array):
        clip_out = self.encoder_tokenizer(
            text=[diagnostic_report], images=[ecg_image], return_tensors="pt", padding="max_length", max_length=self.max_len, truncation=True
        )
        return {
            "encoder_input_ids": clip_out["input_ids"][0].contiguous(),
            "encoder_attention_mask": clip_out["attention_mask"][0].contiguous(),
            "encoder_pixels": clip_out["pixel_values"][0].contiguous(),
            "ecg_signal": ecg_signal,
        }

    def prepare_siglip_input(self, diagnostic_report: str, ecg_image: Image.Image, ecg_signal: np.array):
        siglip_out = self.encoder_tokenizer(
            text=[diagnostic_report], images=[ecg_image], return_tensors="pt", padding="max_length", max_length=self.max_len, truncation=True
        )
        attention_mask = (siglip_out["input_ids"][0] != 1).int()  # siglip does not have an attention mask?
        return {
            "encoder_input_ids": siglip_out["input_ids"][0].contiguous(),
            "encoder_attention_mask": attention_mask.contiguous(),
            "encoder_pixels": siglip_out["pixel_values"][0].contiguous(),
            "ecg_signal": ecg_signal,
        }

    def prepare_vit_input(self, ecg_image: Image.Image, ecg_signal: np.array):
        vit_out = self.encoder_tokenizer(images=ecg_image, return_tensors="pt")
        mask = torch.rand(size=(1, VISION_ENCODERS[self.args.encoder]["num_patches"])) < 0.75
        return {
            "encoder_pixels": vit_out["pixel_values"][0].contiguous(),
            "encoder_mask": mask[0].contiguous(),
            "ecg_signal": ecg_signal,
        }

    ### ELM TRAINING/EVAL/INFERENCE FUNCTIONS ###
    def slice_continuation(self, prompt_ids: list[int], generated_ids: list[int]) -> list[int]:
        K = len(prompt_ids)
        if len(generated_ids) >= K and generated_ids[:K] == prompt_ids:
            return generated_ids[K:]
        return generated_ids

    def get_response_ranges(self, input_ids: List[int]) -> List[Tuple[int, int]]:
        labels = self.create_labels(input_ids)
        ranges, start = [], None
        for i, lab in enumerate(labels):
            if lab != -100 and start is None:
                start = i
            if lab == -100 and start is not None:
                ranges.append((start, i))
                start = None
        if start is not None:
            ranges.append((start, len(labels)))
        return ranges

    def get_ground_truth_responses(self, input_ids: List[int], ranges: List[Tuple[int, int]]) -> List[str]:
        return [self.llm_tokenizer.decode(input_ids[s:e], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for s, e in ranges]

    def get_generated_response_for_turn(self, prompt_input_ids: list[int], generated_ids: list[int]) -> str:
        wt = HF_LLMS[self.args.llm]["watch_tokens"]
        eos = set(wt["eos_token"].keys() if isinstance(wt["eos_token"], dict) else wt["eos_token"])
        fe = wt.get("final_eos_token", ())
        final_eos = set(fe.keys() if isinstance(fe, dict) else fe)
        cont = self.slice_continuation(prompt_input_ids, generated_ids)
        stop_ids = eos | final_eos
        cut = next((i for i, t in enumerate(cont) if t in stop_ids), len(cont))
        cont = cont[:cut]
        return self.llm_tokenizer.decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

    def create_labels(self, input_ids: list[int]) -> list[int]:
        wt = HF_LLMS[self.args.llm]["watch_tokens"]
        BOS = set(wt["bos_token"].keys() if isinstance(wt["bos_token"], dict) else wt["bos_token"])
        EOS = set(wt["eos_token"].keys() if isinstance(wt["eos_token"], dict) else wt["eos_token"])
        fe = wt.get("final_eos_token", ())
        FINAL_EOS = set(fe.keys() if isinstance(fe, dict) else fe)
        labels = [-100] * len(input_ids)
        i, L = 0, len(input_ids)
        seen_bos = False
        in_resp = False
        START = wt["response_start"]["order"]
        k = len(START)

        while i < L:
            tok = input_ids[i]
            if not seen_bos and tok in BOS:
                seen_bos = True
            if seen_bos and (not in_resp) and k > 0:
                if i + k <= L and input_ids[i : i + k] == START:
                    i += k
                    in_resp = True
                    continue
            if in_resp:
                labels[i] = tok
                if tok in EOS:
                    in_resp = False
            i += 1
        if L and input_ids[-1] in FINAL_EOS:  # ensure a single trailing <eos> is labeled (e.g., Gemma 2's final EOS)
            labels[-1] = input_ids[-1]
        return labels

    def create_attention_mask(self, truncated_padded_input: list[int]) -> list[int]:
        bos_token = next(iter(HF_LLMS[self.args.llm]["watch_tokens"]["bos_token"]))
        start_idx = truncated_padded_input.index(bos_token)
        attention_mask = [0] * len(truncated_padded_input)
        attention_mask[start_idx:] = [1] * (len(truncated_padded_input) - start_idx)
        return attention_mask

    def pad_input(self, tokens: list) -> list:
        padding_len = self.args.llm_input_len - len(tokens)
        return [self.llm_tokenizer.pad_token_id] * padding_len + tokens  # left side padding

    def make_prompt(
        self,
        text: str,
    ):
        formatted_text = self.format_text(text)
        prompt = self.chat_template.copy()
        turn_count = 0

        if self.args.no_signal:
            signal_token_placeholders = ""
        else:
            signal_token_placeholders = " ".join([SIGNAL_TOKEN_PLACEHOLDER] * self.args.num_encoder_tokens) + "\n"

        for turn in formatted_text:
            if self.args.dev and is_main():
                print("turn", turn)

            is_human = turn["from"].lower() in ["human", "user"]
            role = prompt.roles[0] if is_human else prompt.roles[1]
            message_value = self.clean_text(turn["value"])

            if is_human and turn_count == 0:
                message_value = f"{signal_token_placeholders}{message_value}"
                turn_count += 1

            prompt.append_message(role, message_value)

        return prompt.get_prompt()

    def find_signal_token_indices(self, input_ids: list[int]) -> list[int]:
        signal_token_id = self.llm_tokenizer.convert_tokens_to_ids(SIGNAL_TOKEN_PLACEHOLDER)
        indices = [i for i, x in enumerate(input_ids) if x == signal_token_id]
        if not indices:
            if self.args.dev and is_main():
                print(f"Signal token ID {signal_token_id} not found in input IDs.")
            return [-1]
        return indices

    def split_prompt(self, prompt: str) -> Tuple[str, str]:
        splitted_prompt = prompt.split(SIGNAL_TOKEN_PLACEHOLDER, 1)
        before = self.llm_tokenizer.encode(splitted_prompt[0], add_special_tokens=False)
        after = self.llm_tokenizer.encode(splitted_prompt[1], add_special_tokens=False)
        return before, after

    def clean_text(self, message_value: str) -> str:
        message_value = TAG_RE.sub("", message_value)
        message_value = IMAGE_WORD_RE.sub(case_preserving_signal, message_value)
        message_value = LEADING_PREFIX_RE.sub("", message_value)
        return message_value

    def format_text(self, text: str):
        if "pretrain-mimic" in self.args.data:
            question = text[0]["value"].replace("\n", "").replace("<ecg>", "")
            answer = text[1]["value"]
            return [
                {"from": "human", "value": question},
                {"from": HF_LLMS[self.args.llm]["role"], "value": answer},
            ]

        if "ecg-qa" in self.args.data:
            _, question, answer = text
            if isinstance(answer, list):
                answer = " ".join(answer)
            return [
                {"from": "human", "value": question},
                {"from": HF_LLMS[self.args.llm]["role"], "value": answer},
            ]
        return text

    def make_chat_template(
        self,
    ):
        chat_template = get_conv_template(HF_LLMS[f"{self.args.llm}"]["chat_template"])
        if HF_LLMS[self.args.llm]["system_prompt"]:
            system_prompt = self.get_system_prompt()
            chat_template.set_system_message(system_prompt)
        return chat_template

    def get_system_prompt(
        self,
    ):
        with open(self.args.system_prompt, encoding="utf-8") as file:
            return file.read()

    ### SIGNAL FUNCTIONS ###
    def normalize(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        min_vals = np.min(ecg_signal)
        max_vals = np.max(ecg_signal)
        normalized = (ecg_signal - min_vals) / (max_vals - min_vals + self.normalize_epsilon)
        clipped_normalized = np.clip(normalized, 0, 1)
        return clipped_normalized, (min_vals, max_vals)

    def blackout_ecg(self, signal):
        return np.zeros_like(signal)

    def noise_ecg(self, signal):
        if random.random() < 0.5:
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

    ### DEBUGGING FUNCTIONS ###
    def decode_and_print_mapping(self, truncated_padded_input: list[int]) -> None:
        tokens = self.llm_tokenizer.convert_ids_to_tokens(truncated_padded_input)
        decoded = self.llm_tokenizer.decode(truncated_padded_input, skip_special_tokens=False)

        print("=== ECG Token Mapping ===")
        for tid, tok in zip(truncated_padded_input, tokens):
            print(f"ID {tid:<6} | Token {tok}")

        print("\n=== Full Decoded String ===")
        print(decoded)

    def check_labels(self, labels):
        labels_np = np.array(labels)
        non_neg_indices = np.where(labels_np != -100)[0]
        if len(non_neg_indices) > 0:
            non_neg_values = labels_np[non_neg_indices].tolist()
            tokens = self.llm_tokenizer.convert_ids_to_tokens(non_neg_values)
            for idx, (token, token_id) in enumerate(zip(tokens, non_neg_values)):
                print(f"{idx}: {token} -> {token_id}")
        else:
            print("No valid labels found (all are -100)")
        print("=" * 100)

    def check_attention_mask(self, truncated_padded_input: list[int], attention_mask: list[int]) -> None:
        tokens = self.llm_tokenizer.convert_ids_to_tokens(truncated_padded_input)
        print("=== Attention Mask Debug ===")
        for i, (tid, tok, mask) in enumerate(zip(truncated_padded_input, tokens, attention_mask)):
            flag = "✓" if mask == 1 else "·"
            print(f"{i:03}: {tid:<6} | {tok:<20} | mask={mask} {flag}")
        print("=" * 100)

    def assert_range_alignment(self, input_ids: List[int], ranges: List[Tuple[int, int]]) -> None:
        wt = HF_LLMS[self.args.llm]["watch_tokens"]
        START = wt["response_start"]["order"]
        EOS = set(wt["eos_token"].keys() if isinstance(wt["eos_token"], dict) else wt["eos_token"])

        k = len(START)
        for s, e in ranges:
            if not (s >= k and input_ids[s - k : s] == START):
                raise AssertionError(f"Misaligned start at {s}: expected response_start immediately before content.")
            if not (e - 1 >= 0 and input_ids[e - 1] in EOS):
                raise AssertionError(f"Missing per-turn EOS at end index {e}.")
