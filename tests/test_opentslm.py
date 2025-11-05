### Taken from https://github.com/mlt94/synchrony/blob/a47eba01f0a9baf4bd5a3d34ed0e485d306d87e4/opentslm/src/test_simple_generation.py
import sys
import os
import torch
import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "OpenTSLM", "src")))

from prompt.full_prompt import FullPrompt
from prompt.text_prompt import TextPrompt

from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from model.llm.OpenTSLMSP import OpenTSLMSP


print("=" * 60)
print("TESTING SIMPLE GENERATION")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
BASE_LLM_ID = "meta-llama/Llama-3.2-3B"

CHECKPOINT_REPO_ID = "OpenTSLM/llama-3.2-3b-ecg-sp"
CHECKPOINT_FILENAME = "softprompt-llama_3_2_3b-ecg.pt"

try:
    print(f"Initializing model architecture using base: {BASE_LLM_ID}...")
    model = OpenTSLMSP(device=device, llm_id=BASE_LLM_ID)
    print("Model architecture built.")

    from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder

    print("Replacing encoder with max_patches=1024 to match checkpoint...")
    model.encoder = TransformerCNNEncoder(max_patches=1024).to(device)

    print("Enabling LoRA...")
    model.enable_lora()

    print(f"Downloading checkpoint from {CHECKPOINT_REPO_ID}...")
    checkpoint_path = hf_hub_download(repo_id=CHECKPOINT_REPO_ID, filename=CHECKPOINT_FILENAME)

    model.load_from_file(checkpoint_path)

    model.eval()


except Exception as e:
    print(f"\n❌ An error occurred: {e}")

text = "This is some sine data sampled at ~100Hz"
ts = np.sin(np.linspace(-np.pi, np.pi, 1000))
print(ts.shape)
TSPrompt = TextTimeSeriesPrompt(text, ts)
prompt = FullPrompt(TextPrompt("Please analyze this time series data"), [TSPrompt], TextPrompt("Now offer your description"))

test_configs = [
    {"max_new_tokens": 500, "name": "500 tokens"},
]
# print("Sample keys:", dataset[0].keys())
# print("Patient ID:", dataset[0].get("patient_id"))
# print("Therapist ID:", dataset[0].get("therapist_id"))
# print("Interview type:", dataset[0].get("interview_type"))

for config in test_configs:
    print("=" * 60)
    print(f"Testing with {config['name']}")
    print("=" * 60)

    with torch.no_grad():
        output = model.eval_prompt(prompt, max_new_tokens=config["max_new_tokens"])

    print(f"Output length: {len(output)} characters")
    print(f"Output: '{output}'")
