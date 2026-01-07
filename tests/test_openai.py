from openai import AsyncOpenAI
import os
import asyncio
import torch
from tqdm.asyncio import tqdm
import numpy as np
from datasets import load_dataset
import ecg_plot
from ecg_bench.utils.file_manager import FileManager
from ecg_bench.runners.elm_evaluator import evaluate_strings
import base64
import sys


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
    **ECG Signal Analysis**
    - Given the ECG signal plot and question, provide a concise answer to the question.
    **Response Style**
    - Deliver responses that are **precise and concise**.
    - Make sure to provide only the answer, no other text or explanations. For example, if the answer is "Yes", just simply respond with "Yes".
    """
    content = [{"type": "input_text", "text": question}]
    if base64_png:
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{base64_png}"})
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


def print_results(all_refs, all_hyps, device):
    if len(all_refs) == 0:
        print("\nNo results collected.")
        return
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


def prepare_sample(data, step, file_manager):
    ecg_path = data["ecg_path"].replace("./data", "./ecg_bench/data")
    ecg_np_file = file_manager.open_npy(ecg_path)
    ecg_signal = ecg_np_file["ecg"]
    question_type, question, answer = data["text"]
    if isinstance(answer, list):
        answer = " ".join(answer)
    ecg_image_path = plot_2d_ecg(ecg_signal, f"ecg_signal_{step}", "./pngs/", 250)
    base64_png = encode_image_to_base64(ecg_image_path)
    messages = build_messages(question, base64_png)
    return {"messages": messages, "answer": answer, "step": step}


async def call_api(client, model, messages):
    for attempt in range(3):
        try:
            response = await client.responses.create(model=model, input=messages)
            return response.output_text
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2**attempt)
            else:
                raise e


async def main():
    FILE_MANAGER = FileManager()
    client = AsyncOpenAI()
    model = "gpt-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_concurrent = 100
    max_samples = 20000

    print(f"Using device: {device}")
    dataset = load_dataset("willxxy/ecg-qa-ptbxl-250-2500", split="fold1_train").with_transform(FILE_MANAGER.decode_batch)

    print("Preparing samples...")
    prepared = []
    for step, data in enumerate(dataset):
        if step >= max_samples:
            break
        try:
            prepared.append(prepare_sample(data, step, FILE_MANAGER))
        except Exception as e:
            print(f"Error preparing step {step}: {e}")

    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def process(sample):
        async with semaphore:
            try:
                output = await call_api(client, model, sample["messages"])
                return {"output": output.lower(), "answer": sample["answer"].lower()}
            except Exception as e:
                print(f"Error at step {sample['step']}: {e}")
                return None

    print(f"Processing {len(prepared)} samples...")
    tasks = [process(s) for s in prepared]
    try:
        for coro in tqdm.as_completed(tasks, total=len(tasks)):
            result = await coro
            if result:
                results.append(result)
    finally:
        all_refs = [r["answer"] for r in results]
        all_hyps = [r["output"] for r in results]
        print_results(all_refs, all_hyps, device)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted!")
        sys.exit(0)
