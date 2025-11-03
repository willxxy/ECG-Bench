import numpy as np
import torch

from ecg_bench.dataloaders.base_dataloader import BaseECGDataset
from ecg_bench.utils.gpu_setup import is_main
from ecg_bench.configs.constants import ECG_TOKEN_PREFIX


class ECGTokenDataset(BaseECGDataset):
    def __init__(self, data, mode, llm_tokenizer_components, args):
        super().__init__(data, mode, args)
        self.llm_tokenizer = llm_tokenizer_components["llm_tokenizer"]
        self.build_ecg_byte()

    def build_ecg_byte(
        self,
    ):
        from ecg_bench.ecg_tokenizers.build_ecg_tokenizers import BuildECGByte

        self.ecg_byte_builder = BuildECGByte(self.args, self.mode)
        new_vocab = [f"{ECG_TOKEN_PREFIX}{ids!s}" for ids in list(self.ecg_byte_builder.vocab.keys())]
        if self.args.dev and is_main():
            print("Length of new tokens", len(new_vocab))
        self.llm_tokenizer.add_tokens(new_vocab)

        if self.args.encoder == "signal2vec":
            self.ecg_token_ids = set(self.llm_tokenizer.convert_tokens_to_ids(new_vocab))

    def __getitem__(self, index):
        instance = self.data[index]
        ecg_path = instance["ecg_path"].replace("./data", "./ecg_bench/data")
        ecg_signal = self.fm.open_npy(ecg_path)["ecg"]

        ### PERTURBATIONS ###
        if self.args.noise_ecg:
            ecg_signal = self.noise_ecg(ecg_signal)
        if self.args.blackout_ecg:
            ecg_signal = self.blackout_ecg(ecg_signal)

        ### PREPARE ECG INPUT ###
        symbols, _ = self.ecg_byte_builder.ecg_to_symbol(ecg_signal)
        ecg_tokens = self.ecg_byte_builder.encode(symbols)
        ecg_tokens = self.llm_tokenizer.convert_tokens_to_ids([f"{ECG_TOKEN_PREFIX}{ids}" for ids in ecg_tokens])

        ### PREPARE TEXT INPUTS ###
        text = instance["text"]
        prompt = self.make_prompt(text)
        if self.args.dev and is_main():
            print("prompt\n", prompt)

        if self.mode == "train":
            return self.prepare_training_set(ecg_tokens, prompt)
        elif self.mode in ["eval", "inference"]:
            return self.prepare_eval_inference_set(ecg_tokens, prompt)

    ### PREPARE TRAINING/EVAL/INFERENCE SETS ###
    def prepare_training_set(
        self,
        ecg_tokens: np.array,
        prompt: str,
    ):
        truncated_padded_input = self.trunc_pad_input(ecg_tokens, prompt)
        attention_mask = self.create_attention_mask(truncated_padded_input)
        labels = self.create_labels(truncated_padded_input)
        ecg_token_indices = self.find_ecg_token_indices(truncated_padded_input)
        if self.args.dev and is_main():
            self.decode_and_print_mapping(truncated_padded_input)
            self.check_labels(labels)
            self.check_attention_mask(truncated_padded_input, attention_mask)

        assert len(truncated_padded_input) == len(attention_mask) == len(labels) == self.args.llm_input_len, (
            f"Length mismatch: {len(truncated_padded_input)} != {len(attention_mask)} != {len(labels)} != {self.args.llm_input_len}"
        )
        return {
            "elm_input_ids": torch.tensor(truncated_padded_input, dtype=torch.int64),
            "elm_labels": torch.tensor(labels, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "ecg_token_indices": torch.tensor(ecg_token_indices, dtype=torch.int64),
        }

    def prepare_eval_inference_set(
        self,
        ecg_tokens: np.array,
        prompt: str,
    ):
        truncated_padded_input = self.trunc_pad_input(ecg_tokens, prompt)
        attention_mask = self.create_attention_mask(truncated_padded_input)
        ecg_token_indices = self.find_ecg_token_indices(truncated_padded_input)
        assert len(truncated_padded_input) == len(attention_mask), f"Length mismatch: {len(truncated_padded_input)} != {len(attention_mask)}"
        return {
            "elm_input_ids": torch.tensor(truncated_padded_input, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "ecg_token_indices": torch.tensor(ecg_token_indices, dtype=torch.int64),
        }

    ### PADDING/TRUNCATION FUNCTIONS ###
    def trunc_pad_input(self, ecg_tokens: np.ndarray, prompt: str):
        before, after = self.split_prompt(prompt)
        if self.mode in ["eval", "inference"]:
            return before + ecg_tokens + after
        else:
            min_ecg_token_len = int(self.args.min_ecg_tokens_len)

            before_len, after_len, ecg_token_len = len(before), len(after), len(ecg_tokens)

            if before_len + after_len + ecg_token_len == self.args.llm_input_len:
                return before + ecg_tokens + after
            elif before_len + after_len + ecg_token_len < self.args.llm_input_len:
                return self.pad_input(before + ecg_tokens + after)

            if before_len + min_ecg_token_len > self.args.llm_input_len:
                raise ValueError("before + min_ecg exceeds llm_input_len; lower min_ecg_tokens_len.")

            target_ecg = min(ecg_token_len, max(min_ecg_token_len, self.args.llm_input_len - (before_len + after_len)))
            ecg_tokens = ecg_tokens[:target_ecg]

            remaining_after = self.args.llm_input_len - before_len - len(ecg_tokens)
            after = after[: max(remaining_after, 0)]

            return before + ecg_tokens + after

    def find_ecg_token_indices(self, input_ids: list[int]) -> list[int]:
        ecg_token_indices = [i for i, tid in enumerate(input_ids) if tid in self.ecg_token_ids]
        if not ecg_token_indices or self.args.encoder != "signal2vec":
            if self.args.dev and is_main():
                print("No ECG tokens found in input_ids.")
            return [-1]
        return ecg_token_indices
