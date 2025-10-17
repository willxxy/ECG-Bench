import numpy as np
from imgaug import augmenters as iaa
from PIL import Image
import random
import torch
from typing import Optional

from ecg_bench.dataloaders.base_dataloader import BaseECGDataset
from ecg_bench.utils.viz import VizUtil
from ecg_bench.utils.gpu_setup import is_main


class ECGImageDataset(BaseECGDataset):
    def __init__(self, data, mode, llm_tokenizer_components, encoder_tokenizer_components, args):
        super().__init__(data, mode, args)
        self.llm_tokenizer = llm_tokenizer_components["llm_tokenizer"] if llm_tokenizer_components else None
        self.encoder_tokenizer = encoder_tokenizer_components["encoder_tokenizer"]
        self.viz = VizUtil()

    def __getitem__(self, index):
        instance = self.data[index]
        ecg_path = instance["ecg_path"]
        ecg_path = ecg_path.replace("./data", "./ecg_bench/data")  # for my own local
        ecg_np_file = self.fm.open_npy(ecg_path)
        ecg_signal = ecg_np_file["ecg"]
        self.viz.plot_2d_ecg(ecg_signal, "ecg_signal", "./ecg_signal.png", 250)
        diagnostic_report = ecg_np_file["report"]

        ### PERTURBATIONS ###
        if self.args.noise_ecg:
            ecg_signal = self.noise_ecg(ecg_signal)
        if self.args.blackout_ecg:
            ecg_signal = self.blackout_ecg(ecg_signal)
        ecg_image = self.signal_to_image(ecg_signal)

        if self.encoder_tokenizer is not None:
            if self.args.encoder == "clip-vit-base-patch32":
                encoder_tokenizer_out = self.prepare_clip_input(diagnostic_report, ecg_image, ecg_signal)
            elif self.args.encoder == "siglip-base-patch16-224":
                encoder_tokenizer_out = self.prepare_siglip_input(diagnostic_report, ecg_image, ecg_signal)
            elif self.args.encoder == "vit-base-patch16-224-in21k":
                encoder_tokenizer_out = self.prepare_vit_input(ecg_image, ecg_signal)
        else:
            encoder_tokenizer_out = {"ecg_signal": ecg_signal}

        text = instance["text"]
        if self.args.llm:
            prompt = self.make_prompt(text)
            if self.args.dev and is_main():
                print("prompt\n", prompt)
        else:
            prompt = None

        if self.mode == "train":
            return self.prepare_training_set(prompt, encoder_tokenizer_out)
        elif self.mode in ["eval", "inference"]:
            return self.prepare_eval_inference_set(prompt, encoder_tokenizer_out)

    def prepare_training_set(
        self,
        prompt: Optional[str],
        encoder_tokenizer_out: dict,
    ):
        if prompt is not None:
            truncated_padded_input = self.trunc_pad_input(prompt)
            signal_id_indices = self.find_signal_token_indices(truncated_padded_input)
            attention_mask = self.create_attention_mask(truncated_padded_input)
            labels = self.create_labels(truncated_padded_input)
            assert len(truncated_padded_input) == len(attention_mask) == len(labels) == self.args.llm_input_len, (
                f"Length mismatch: {len(truncated_padded_input)} != {len(attention_mask)} != {len(labels)} != {self.args.llm_input_len}"
            )
            elm = {
                "elm_input_ids": torch.tensor(truncated_padded_input, dtype=torch.int64),
                "elm_labels": torch.tensor(labels, dtype=torch.int64),
                "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
                "signal_id_indices": torch.tensor(signal_id_indices, dtype=torch.int64),
            }
            return {**elm, **encoder_tokenizer_out}
        else:
            return encoder_tokenizer_out

    def prepare_eval_inference_set(
        self,
        prompt: Optional[str],
        encoder_tokenizer_out: dict,
    ):
        if prompt is not None:
            truncated_padded_input = self.trunc_pad_input(prompt)
            signal_id_indices = self.find_signal_token_indices(truncated_padded_input)
            attention_mask = self.create_attention_mask(truncated_padded_input)
            assert len(truncated_padded_input) == len(attention_mask), f"Length mismatch: {len(truncated_padded_input)} != {len(attention_mask)}"
            elm = {
                "elm_input_ids": torch.tensor(truncated_padded_input, dtype=torch.int64),
                "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
                "signal_id_indices": torch.tensor(signal_id_indices, dtype=torch.int64),
            }
            return {**elm, **encoder_tokenizer_out}
        else:
            return encoder_tokenizer_out

    ### SIGNAL TO IMAGE FUNCTIONS ###
    def signal_to_image(self, ecg_signal: np.array):
        image = self.viz.get_plot_as_image(ecg_signal, 250)  # 250 Hz
        if self.args.augment_ecg_image and random.random() < 0.6:
            return self.augment_image(image)
        return Image.fromarray(image)

    ### IMAGE AUGMENTATION FUNCTIONS ###
    def augment_image(self, image: np.array):
        seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-5, 5))),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 1.5))),
            iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-30, 30), per_channel=True)),
        ])
        augmented_image = seq.augment_image(image)
        return Image.fromarray(augmented_image)

    def trunc_pad_input(self, prompt: str):
        prompt_tokens = self.llm_tokenizer.encode(prompt, add_special_tokens=False)
        if self.mode in ["eval", "inference"]:
            return prompt_tokens
        else:
            prompt_len = len(prompt_tokens)
            if prompt_len == self.args.llm_input_len:
                return prompt_tokens
            elif prompt_len < self.args.llm_input_len:
                return self.pad_input(prompt_tokens)
            truncated_prompt = prompt_tokens[-self.args.llm_input_len :]
            return truncated_prompt
