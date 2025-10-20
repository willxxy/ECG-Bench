from typing import Literal
import torch
import re

# Main arg MODE types
Mode = Literal["train", "eval", "inference", "post_train", "ecg_tokenizer", "preprocess", "rag", "signal2vec"]

# Directories
RUNS_DIR = "./ecg_bench/runs"
DATA_DIR = "./ecg_bench/data"
CONFIG_DIR = "./ecg_bench/configs"

# Datasets
BASE_DATASETS = ["ptb", "mimic", "code15", "cpsc", "csn"]
MAPPED_DATASETS = [
    "ecg_bench_pulse",
    "ecg_instruct_pulse",
    "pretrain_mimic",
    "ecg_instruct_45k",
    "ecg-qa_ptbxl",
    "ecg-qa_mimic-iv-ecg",
    "ecg_grounding_pulse",
    "ecg_grounding",
    "ecg_grounding_test",
]

# Hugging Face
HF_CACHE_DIR = "./.huggingface"
HF_DATASETS = [
    # ECG-QA MIMIC-IV-ECG (https://github.com/Jwoo5/ecg-qa)
    "ecg-qa-mimic-iv-ecg-250-500",
    "ecg-qa-mimic-iv-ecg-250-1250",
    "ecg-qa-mimic-iv-ecg-250-2500",
    # ECG-QA PTB-XL (https://github.com/Jwoo5/ecg-qa)
    "ecg-qa-ptbxl-250-500",
    "ecg-qa-ptbxl-250-1250",
    "ecg-qa-ptbxl-250-2500",
    # Pretrain MIMIC (https://github.com/YubaoZhao/ECG-Chat)
    "pretrain-mimic-250-500",
    "pretrain-mimic-250-1250",
    "pretrain-mimic-250-2500",
    # ECG Instruct 45k (https://github.com/YubaoZhao/ECG-Chat)
    "ecg-instruct-45k-250-500",
    "ecg-instruct-45k-250-1250",
    "ecg-instruct-45k-250-2500",
    # ECG Bench Pulse (https://github.com/AIMedLab/PULSE)
    "ecg-bench-pulse-250-500",
    "ecg-bench-pulse-250-1250",
    "ecg-bench-pulse-250-2500",
    # ECG Instruct Pulse (https://github.com/AIMedLab/PULSE)
    "ecg-instruct-pulse-250-500",
    "ecg-instruct-pulse-250-1250",
    "ecg-instruct-pulse-250-2500",
]

HF_LLMS = {
    "llama-3.2-1b-instruct": {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "tokenizer": "meta-llama/Llama-3.2-1B-Instruct",
        "chat_template": "llama-3",
        "native_dtype": torch.bfloat16,
        "tokens_to_add": {
            "additional_special_tokens": [],
        },
        "find_unused_parameters": False,
        "model_hidden_size": None,
        "system_prompt": True,
        "role": "assistant",
        "watch_tokens": {
            "bos_token": {128000: "<|begin_of_text|>"},
            "eos_token": {128009: "<|eot_id|>"},
            "response_start": {
                "order": [128006, 78191, 128007, 271],
                128006: "<|start_header_id|>",
                78191: "assistant",
                128007: "<|end_header_id|>",
                271: "ĊĊ",
            },
        },
    },
    "qwen2.5-1.5b-instruct": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct",
        "chat_template": "qwen-7b-chat",
        "native_dtype": torch.bfloat16,
        "tokens_to_add": {
            "additional_special_tokens": [],
        },
        "find_unused_parameters": False,
        "model_hidden_size": None,
        "system_prompt": True,
        "role": "assistant",
        "watch_tokens": {
            "bos_token": {151644: "<|im_start|>"},
            "eos_token": {151645: "<|im_end|>"},
            "response_start": {
                "order": [151645, 198, 151644, 77091, 198],
                151645: "<|im_end|>",
                198: "Ċ",
                151644: "<|im_start|>",
                77091: "assistant",
            },
        },
    },
    "gemma-2-2b-it": {
        "model": "google/gemma-2-2b-it",
        "tokenizer": "google/gemma-2-2b-it",
        "chat_template": "gemma-2",
        "native_dtype": torch.bfloat16,
        "tokens_to_add": {
            "additional_special_tokens": [],
        },
        "find_unused_parameters": False,
        "model_hidden_size": None,
        "system_prompt": False,
        "role": "model",
        "watch_tokens": {
            "bos_token": {2: "<bos>"},
            "eos_token": {107: "<end_of_turn>"},
            "final_eos_token": {1: "<eos>"},  # this is the eos but this eos goes at the very end. not used per turn
            "response_start": {
                "order": [107, 108, 106, 2516, 108],
                107: "<end_of_turn>",
                108: "",  # sort of like Ċ but not printed.
                106: "<start_of_turn>",
                2516: "model",
            },
        },
    },
}

VISION_ENCODERS = {
    "clip-vit-base-patch32": {
        "model": "openai/clip-vit-base-patch32",
        "tokenizer": "openai/clip-vit-base-patch32",
        "find_unused_parameters": False,
        "strict": True,
        "model_hidden_size": None,
        "projection_dim": None,
        "encoder_input_len": 77,
    },
    "siglip-base-patch16-224": {
        "model": "google/siglip-base-patch16-224",
        "tokenizer": "google/siglip-base-patch16-224",
        "find_unused_parameters": False,
        "strict": True,
        "model_hidden_size": None,
        "projection_dim": None,
        "encoder_input_len": 64,
    },
    "vit-base-patch16-224-in21k": {
        "model": "google/vit-base-patch16-224-in21k",
        "tokenizer": "google/vit-base-patch16-224-in21k",
        "find_unused_parameters": False,
        "strict": True,
        "model_hidden_size": None,
        "projection_dim": None,
        "num_patches": None,
        "encoder_input_len": None,
    },
}

VISION_ENCODERS_INPUT_MAPPING = {
    "clip-vit-base-patch32": {
        "input_ids": "encoder_input_ids",
        "attention_mask": "encoder_attention_mask",
        "pixel_values": "encoder_pixels",
        "output_hidden_states": True,
    },
    "siglip-base-patch16-224": {
        "input_ids": "encoder_input_ids",
        "attention_mask": "encoder_attention_mask",
        "pixel_values": "encoder_pixels",
        "output_hidden_states": True,
    },
    "vit-base-patch16-224-in21k": {
        "pixel_values": "encoder_pixels",
        "bool_masked_pos": "encoder_mask",
        "output_hidden_states": True,
    },
}


## Token ID
SIGNAL_TOKEN_PLACEHOLDER = "<signal>"
ECG_TOKEN_PREFIX = "signal_"

# Encoders
ECG_ENCODERS = {
    "merl": {
        "model": "resnet101",
        "tokenizer": "ncbi/MedCPT-Query-Encoder",
        "find_unused_parameters": True,
        "strict": False,
        "model_hidden_size": 256,
        "projection_dim": 2048,
        "spacial_dim": None,
        "encoder_input_len": 64,
    },
    "st_mem": {
        "find_unused_parameters": False,
        "strict": False,
        "model_hidden_size": 768,
        "projection_dim": 256,
        "encoder_input_len": None,
    },
    "mtae": {
        "find_unused_parameters": False,
        "strict": False,
        "model_hidden_size": 768,
        "projection_dim": 256,
        "encoder_input_len": None,
    },
    "mlae": {
        "find_unused_parameters": False,
        "strict": False,
        "model_hidden_size": 768,
        "projection_dim": 256,
        "encoder_input_len": None,
    },
    "projection": {
        "find_unused_parameters": False,
        "strict": False,
        "model_hidden_size": None,
        "projection_dim": None,
        "encoder_input_len": None,
    },
}


# TEXT CLEANER
LEADING_PREFIX_RE = re.compile(
    r"^\s*(?:[:：]\s*|(?:user|assistant|human|gpt|model|system|q|a)\s*:\s*)+",
    flags=re.IGNORECASE,
)
TAG_RE = re.compile(r"<\s*(?:ecg|image)\s*>\s*\n?", flags=re.IGNORECASE)
IMAGE_WORD_RE = re.compile(r"\b(image)\b", flags=re.IGNORECASE)


def case_preserving_signal(m: re.Match) -> str:
    w = m.group(1)
    return "Signal" if w[0].isupper() else "signal"
