import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pytest

from ecg_bench.configs.constants import HF_LLMS, HF_DATASETS, HF_CACHE_DIR
from ecg_bench.utils.file_manager import FileManager

FILE_MANAGER = FileManager()

pytestmark = [pytest.mark.transformers]


def test_random_model_loading():
    """Test loading a random transformers model and tokenizer from HF_LLMS."""
    model_name = random.choice(list(HF_LLMS.keys()))
    model_config = HF_LLMS[model_name]

    print(f"Testing model: {model_name}")
    print(f"Model ID: {model_config['model']}")
    print(f"Tokenizer ID: {model_config['tokenizer']}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer"],
        cache_dir=HF_CACHE_DIR,
    )

    assert tokenizer is not None
    assert tokenizer.vocab_size > 0

    import torch

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model"],
        dtype=torch.bfloat16,
        cache_dir=HF_CACHE_DIR,
    )

    assert model is not None

    test_text = "Hello, this is a test."
    tokens = tokenizer.encode(test_text)
    assert len(tokens) > 0

    decoded_text = tokenizer.decode(tokens)
    assert isinstance(decoded_text, str)

    print(f"Successfully loaded and tested {model_name}")


def test_random_dataset_loading():
    """Test loading a random dataset from HF_DATASETS."""
    dataset_name = random.choice(HF_DATASETS)

    print(f"Testing dataset: {dataset_name}")

    dataset = load_dataset(f"willxxy/{dataset_name}", split="fold1_train").with_transform(FILE_MANAGER.decode_batch)

    # Verify dataset loaded successfully
    assert dataset is not None
    assert len(dataset) > 0

    # Get a sample from the dataset
    sample = dataset[0]
    assert isinstance(sample, dict)

    # Print basic info about the dataset
    print(f"Dataset size: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print(f"Sample keys: {list(sample.keys())}")

    print(f"Successfully loaded and tested {dataset_name}")
