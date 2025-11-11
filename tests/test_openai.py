from openai import OpenAI
import os
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import ecg_plot
from ecg_bench.utils.file_manager import FileManager
from ecg_bench.utils.viz import VizUtil
from ecg_bench.runners.elm_evaluator import evaluate_strings
import base64


def plot_2d_ecg(ecg: np.ndarray, title: str, save_dir: str, sample_rate: int) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ecg_plot.plot(ecg, title="", sample_rate=sample_rate)
    ecg_plot.save_as_png(file_name=title, path=save_dir)
    return os.path.join(save_dir, f"{title}.png")


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_messages(question: str, base64_png: str | None):
    system_prompt = """
    You are an expert multimodal assistant with advanced knowledge in **clinical cardiac electrophysiology**.

**Input Identification**
- Detect whether the input is **text**, **ECG signals (time-series data)**, or **both**.

**ECG Signal Analysis**
- Treat ECG as cardiac time-series data.
- Provide expert interpretation of **heart rate, rhythm, conduction, arrhythmias, and other electrophysiologic abnormalities**.

**Multimodal Reasoning**
- When both text and ECG are given, integrate them into a unified, **cardiac electrophysiologist-level assessment**.

**Response Style**
- Deliver responses that are **precise, concise, and clinically authoritative**, grounded in electrophysiology and natural language reasoning.
- For general, non-ECG questions, respond as a capable **general assistant**.
- Make sure to provide only the answer, no other text.
    """
    content = [{"type": "input_text", "text": question}]
    if base64_png:
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{base64_png}"})
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


DATASETS = [
    "ecg-qa-ptbxl-250-2500",
    "ecg-qa-mimic-iv-ecg-250-2500",
]
FILE_MANAGER = FileManager()
VIZ_UTIL = VizUtil()

print("=" * 60)
print("TESTING SIMPLE GENERATION")
print("=" * 60)

client = OpenAI()
model = "gpt-5"

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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
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
    ecg_image_path = plot_2d_ecg(ecg_signal, f"ecg_signal_{step}", "./pngs/", 250)
    base64_png = encode_image_to_base64(ecg_image_path)
    messages = build_messages(question, base64_png)
    output = client.responses.create(
        model=model,
        # instructions=system_prompt,
        input=messages,
    )
    print("Question:\n", question)
    print("--------------------------------")
    print("Answer:\n", answer)
    print("--------------------------------")
    print("Output:\n", output.output_text)
    print("################################")
    all_refs.append(answer)
    all_hyps.append(output)
    if step > 20000:
        break

results = evaluate_strings(all_refs, all_hyps, device=device)
print(results)
