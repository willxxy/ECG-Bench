import argparse
from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoProcessor
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from ecg_bench.configs.constants import HF_DATASETS, HF_CACHE_DIR, HF_LLMS, SIGNAL_TOKEN_PLACEHOLDER, ECG_ENCODERS, VISION_ENCODERS
from ecg_bench.utils.gpu_setup import is_main, get_world_size, get_rank


class BuildDataLoader:
    def __init__(
        self,
        mode: str,
        args: argparse.Namespace,
    ):
        self.args = args
        self.mode = mode
        assert self.args.data in HF_DATASETS, print(f"{self.args.data} is not supported.")
        self.assert_data_model_match()

    def build_dataloader(
        self,
    ):
        self.llm_tokenizer_components = None
        self.data = self.load_dataset()
        if self.args.llm:
            self.llm_tokenizer_components = self.build_llm_tokenizer()
        if self.args.encoder:
            self.encoder_tokenizer_components = self.build_encoder_tokenizer()
        self.torch_dataset = self.build_torch_dataset()
        torch_data_loader = self.build_torch_dataloader()
        return torch_data_loader

    ### TORCH DATALOADER FUNCTIONS ###
    def build_torch_dataloader(
        self,
    ):
        self.get_torch_dataloader_sampler()
        if self.mode == "train":
            torch_data_loader = DataLoader(
                self.torch_dataset,
                batch_size=self.args.batch_size,
                shuffle=(self.sampler is None),
                num_workers=2 if self.args.encoder and not self.args.llm else 0,
                sampler=self.sampler,
                pin_memory=True,
                collate_fn=self.collate_fn,
            )
        elif self.mode in ["inference", "eval"]:
            torch_data_loader = DataLoader(
                self.torch_dataset,
                batch_size=1,  # batched inference/eval not implemented
                shuffle=False,
                pin_memory=True,
                collate_fn=self.collate_fn,
            )
        return torch_data_loader

    def get_torch_dataloader_sampler(
        self,
    ):
        if self.args.distributed:
            self.sampler = DistributedSampler(self.torch_dataset, num_replicas=get_world_size(), rank=get_rank(), seed=self.args.seed, shuffle=True)
        else:
            self.sampler = None

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return {
                "encoder_input_ids": torch.tensor([], dtype=torch.int64),
                "encoder_pixels": torch.tensor([], dtype=torch.float32),
                "encoder_attention_mask": torch.tensor([], dtype=torch.float32),
                "encoder_mask": torch.tensor([], dtype=torch.bool),
                "signal_id_indices": torch.tensor([], dtype=torch.int64),
                "elm_input_ids": torch.tensor([], dtype=torch.int64),
                "elm_labels": torch.tensor([], dtype=torch.int64),
                "elm_attention_mask": torch.tensor([], dtype=torch.float32),
                "ecg_signal": torch.tensor([], dtype=torch.float32),
                "truncated_padded_ecg_tokens": torch.tensor([], dtype=torch.int64),
            }

        if self.args.encoder == "signal2vec":
            pad_id = -2
            pad_fields = ["truncated_padded_ecg_tokens", "signal_id_indices"]
            for field in pad_fields:
                if all(field in item for item in batch):
                    max_len = max(len(item[field]) for item in batch)
                    for item in batch:
                        tensor = item[field]
                        pad_len = max_len - len(tensor)
                        if pad_len > 0:
                            item[field] = torch.cat([tensor, torch.full((pad_len,), pad_id, dtype=torch.int64)])
                        elif pad_len < 0:
                            item[field] = tensor[:max_len]
        return torch.utils.data.dataloader.default_collate(batch)

    ### TORCH DATASET FUNCTIONS ###
    def build_torch_dataset(
        self,
    ):
        if self.args.ecg_token:
            from ecg_bench.dataloaders.ecg_token_dataloader import ECGTokenDataset

            torch_dataset = ECGTokenDataset(self.data, self.mode, self.llm_tokenizer_components, self.args)
        elif self.args.ecg_image:
            from ecg_bench.dataloaders.ecg_image_dataloader import ECGImageDataset

            torch_dataset = ECGImageDataset(self.data, self.mode, self.llm_tokenizer_components, self.encoder_tokenizer_components, self.args)
        elif self.args.ecg_signal:
            from ecg_bench.dataloaders.ecg_signal_dataloader import ECGSignalDataset

            torch_dataset = ECGSignalDataset(self.data, self.mode, self.llm_tokenizer_components, self.encoder_tokenizer_components, self.args)
        elif self.args.ecg_stacked_signal:
            from ecg_bench.dataloaders.ecg_stacked_signal_dataloader import ECGStackedSignalDataset

            torch_dataset = ECGStackedSignalDataset(self.data, self.mode, self.llm_tokenizer_components, self.encoder_tokenizer_components, self.args)
        else:
            raise ValueError("Please choose an input representation.")
        return torch_dataset

    ### HF DATASET FUNCTIONS ###
    def load_dataset(
        self,
    ):
        if self.mode in ["train", "post_train"]:
            data = load_dataset(f"willxxy/{self.args.data}", split=f"fold{self.args.fold}_train", cache_dir=HF_CACHE_DIR).with_transform(
                self.decode_batch
            )
        elif self.mode in ["eval", "inference"]:
            data = load_dataset(f"willxxy/{self.args.data}", split=f"fold{self.args.fold}_test", cache_dir=HF_CACHE_DIR).with_transform(
                self.decode_batch
            )
        if is_main():
            print("Length of Dataset Considered:", len(data))
        return data

    def decode_batch(self, batch: dict) -> dict:
        if "text" in batch:
            out = []
            for t in batch["text"]:
                try:
                    out.append(json.loads(t))
                except Exception:
                    out.append(t)
            batch["text"] = out
        return batch

    ### ENCODER TOKENIZER FUNCTIONS ###
    def build_encoder_tokenizer(
        self,
    ):
        if self.args.encoder == "projection":
            return {"encoder_tokenizer": None}
        elif self.args.encoder == "st_mem":
            return {"encoder_tokenizer": None}
        elif self.args.encoder == "mtae":
            return {"encoder_tokenizer": None}
        elif self.args.encoder == "mlae":
            return {"encoder_tokenizer": None}
        elif self.args.encoder == "merl":
            return {"encoder_tokenizer": AutoTokenizer.from_pretrained(ECG_ENCODERS[self.args.encoder]["tokenizer"], cache_dir=HF_CACHE_DIR)}
        elif self.args.encoder in VISION_ENCODERS:
            return {"encoder_tokenizer": AutoProcessor.from_pretrained(VISION_ENCODERS[self.args.encoder]["tokenizer"], cache_dir=HF_CACHE_DIR)}

    ### HF LLM TOKENIZER FUNCTIONS ###
    def build_llm_tokenizer(
        self,
    ):
        llm_tokenizer = AutoTokenizer.from_pretrained(
            HF_LLMS[self.args.llm]["tokenizer"],
            cache_dir=HF_CACHE_DIR,
        )
        if self.args.dev and is_main():
            print("Before Modification\n")
            self.print_llm_tokenizer_info(llm_tokenizer)
        llm_tokenizer = self.modify_llm_tokenizer(llm_tokenizer)
        if self.args.dev and is_main():
            print("After Modification\n")
            self.print_llm_tokenizer_info(llm_tokenizer)
        return {"llm_tokenizer": llm_tokenizer}

    def modify_llm_tokenizer(self, llm_tokenizer):
        if getattr(llm_tokenizer, "pad_token", None) is None:  # llama 3.2
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        tokens_to_add = HF_LLMS[self.args.llm]["tokens_to_add"]

        if self.args.encoder and self.args.llm:
            tokens_to_add["additional_special_tokens"].append(SIGNAL_TOKEN_PLACEHOLDER)

        llm_tokenizer.add_special_tokens(tokens_to_add)
        return llm_tokenizer

    ### DEV FUNCTIONS ###
    def print_llm_tokenizer_info(self, llm_tokenizer):
        print("Vocab Size:", llm_tokenizer.vocab_size)
        print("special_tokens_map:", llm_tokenizer.special_tokens_map)
        print("all_special_tokens:", llm_tokenizer.all_special_tokens)
        print("all_special_ids:", llm_tokenizer.all_special_ids)
        for k in ("pad", "bos", "eos", "unk"):
            t = getattr(llm_tokenizer, f"{k}_token", None)
            i = getattr(llm_tokenizer, f"{k}_token_id", None)
            print(f"{k.upper()} -> token: {t!r}, id: {i}")
        print("-" * 20)

    def assert_data_model_match(self):
        if self.args.ecg_token:
            assert self.args.encoder is None or self.args.encoder == "signal2vec", print("ecg_token mode should not specify encoder")
        elif self.args.ecg_image or self.args.ecg_stacked_signal:
            assert self.args.encoder in VISION_ENCODERS, print(f"ecg_image/ecg_stacked_signal requires vision encoder, got {self.args.encoder}")
        elif self.args.ecg_signal:
            assert self.args.encoder in ECG_ENCODERS, print(f"ecg_signal requires ECG encoder, got {self.args.encoder}")
