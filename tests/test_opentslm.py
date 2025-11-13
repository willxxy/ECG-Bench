### Taken from https://github.com/mlt94/synchrony/blob/a47eba01f0a9baf4bd5a3d34ed0e485d306d87e4/opentslm/src/test_simple_generation.py
import sys
import os
import re
import torch
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import ecg_plot
from ecg_bench.utils.file_manager import FileManager
from ecg_bench.utils.viz import VizUtil
from ecg_bench.runners.elm_evaluator import evaluate_strings


def plot_2d_ecg(ecg: np.ndarray, title: str, save_dir: str, sample_rate: int, lead_index) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ecg_plot.plot(ecg, title="", sample_rate=sample_rate)
    ecg_plot.save_as_png(file_name=title, path=save_dir)
    return os.path.join(save_dir, f"{title}.png")


DATASETS = [
    "ecg-qa-ptbxl-250-2500",
    "ecg-qa-mimic-iv-ecg-250-2500",
]
FILE_MANAGER = FileManager()
VIZ_UTIL = VizUtil()
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

test_configs = [
    {"max_new_tokens": 256, "name": "256 tokens"},
]


output_img = False
lead_names = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]
dataset = load_dataset(f"willxxy/{DATASETS[0]}", split="fold1_train").with_transform(FILE_MANAGER.decode_batch)
all_refs, all_hyps = [], []
for step, data in enumerate(tqdm(dataset)):
    ecg_path = data["ecg_path"]
    ecg_path = ecg_path.replace("./data", "./ecg_bench/data")
    ecg_np_file = FILE_MANAGER.open_npy(ecg_path)
    ecg_signal = ecg_np_file["ecg"]
    question_type, question, answer = data["text"]
    if isinstance(answer, list):
        answer = " ".join(answer)

    ts_prompt_list = []
    for i in range(ecg_signal.shape[0]):
        mean_val = float(np.mean(ecg_signal[i]))
        std_val = float(np.std(ecg_signal[i]))
        if std_val > 1e-6:  # Avoid division by zero
            normalized_signal = (ecg_signal[i] - mean_val) / std_val
        else:
            normalized_signal = ecg_signal[i] - mean_val
        text = f"ECG Lead {lead_names[i]}"
        text += f" - sampled at ~250Hz, normalized (mean={mean_val:.3f}, std={std_val:.3f})"
        ts_prompt_list.append(TextTimeSeriesPrompt(text, normalized_signal.tolist()))

    pre_prompt_cot = f"""You are an expert cardiologist analyzing an ECG (electrocardiogram).

Clinical Context: 12-lead ECG recording.

Your task is to examine the ECG signal and answer the following medical question:

Question: {question}

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding the cardiac condition.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate to the question."""

    post_prompt_cot = """
Based on your analysis of the ECG data, provide your answer.
Make sure that your last word is the answer. You MUST end your response with "Answer: "
"""
    post_prompt_cot = post_prompt_cot.strip()
    prompt = FullPrompt(TextPrompt(pre_prompt_cot), ts_prompt_list, TextPrompt(post_prompt_cot))

    with torch.no_grad():
        output = model.eval_prompt(prompt, max_new_tokens=test_configs[0]["max_new_tokens"])
    if output_img:
        VIZ_UTIL.plot_2d_ecg(ecg_signal, f"ecg_signal_{step}", "./pngs/", 250)
    # print("Question:\n", question)
    # print("--------------------------------")
    # print("Answer:\n", answer)
    # print("--------------------------------")
    output = (
        (re.findall(r"Answer:\s*(.+?)(?=\n(?:[-#]{3,}|Output:|Question:)|\Z)", output, re.DOTALL) or [""])[-1].strip().replace("\n", " ").rstrip(" .")
    )
    # print("Output:\n", output)
    # print("################################")
    all_refs.append(answer)
    all_hyps.append(output)
    if step > 20000:
        break

results = evaluate_strings(all_refs, all_hyps, device=device)
print("\n=== N-Turn Evaluation (generated vs. gold response only) ===")
print(f"Pairs: {len(all_refs)}")
print(f"ACC: {results['ACC']:.4f}")
print(f"BLEU (corpus): {results['BLEU']:.4f}")
print(f"BLEU_sent (effective): {results['BLEU_sent']:.4f}")
print(f"METEOR: {results['METEOR']:.4f}")
r = results["ROUGE"]
print(f"ROUGE-1/2/L (F): {r['rouge-1']:.4f} / {r['rouge-2']:.4f} / {r['rouge-l']:.4f}")
print(
    f"BERTScore (mean) P/R/F1: "
    f"{float(np.mean(results['BERTSCORE']['hf-prec'])):.4f} / "
    f"{float(np.mean(results['BERTSCORE']['hf-rec'])):.4f} / "
    f"{float(np.mean(results['BERTSCORE']['hf-f1'])):.4f}"
)
