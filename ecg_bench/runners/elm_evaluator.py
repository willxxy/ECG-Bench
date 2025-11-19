import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import scipy.stats as stats
from evaluate import load
from ecg_bench.utils.gpu_setup import is_main, train_dev_break
from tqdm import tqdm
import torch


def calculate_acc(references, hypotheses):
    return np.mean([ref == hyp for ref, hyp in zip(references, hypotheses)])


def calculate_bleu(references, hypotheses):
    smoother = SmoothingFunction()
    references = [[r.split()] for r in references]
    hypotheses = [h.split() for h in hypotheses]
    return corpus_bleu(references, hypotheses, smoothing_function=smoother.method1)


def calculate_meteor(references, hypotheses):
    scores = [meteor_score([ref.split()], hyp.split()) for ref, hyp in zip(references, hypotheses)]
    return np.mean(scores)


def calculate_rouge(references, hypotheses):
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }


def calculate_bertscore(references, hypotheses, device):
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=hypotheses,
        references=references,
        lang="en",
        device=device,
    )
    return {
        "hf-prec": results["precision"],
        "hf-rec": results["recall"],
        "hf-f1": results["f1"],
    }


def calculate_bleu_sentence_effective(references, hypotheses):
    sm = SmoothingFunction().method1

    def sent_bleu(r, h):
        r_tok = r.split()
        h_tok = h.split()
        n = min(4, len(r_tok), len(h_tok))
        if n == 0:
            return 0.0
        weights = tuple([1.0 / n] * n)
        return sentence_bleu([r_tok], h_tok, weights=weights, smoothing_function=sm)

    return float(np.mean([sent_bleu(r, h) for r, h in zip(references, hypotheses)]))


def evaluate_strings(references, hypotheses, device):
    if len(references) != len(hypotheses):
        raise ValueError("The number of references and hypotheses must be the same.")
    valid_pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses) if ref and hyp]
    if not valid_pairs:
        return {
            "BLEU": 0,
            "BLEU_sent": 0,
            "METEOR": 0.0,
            "ROUGE": {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0},
            "BERTSCORE": {"hf-prec": [0.0], "hf-rec": [0.0], "hf-f1": [0.0]},
            "ACC": 0.0,
        }
    valid_refs, valid_hyps = zip(*valid_pairs)
    return {
        "BLEU": calculate_bleu(valid_refs, valid_hyps),
        "BLEU_sent": calculate_bleu_sentence_effective(valid_refs, valid_hyps),
        "METEOR": calculate_meteor(valid_refs, valid_hyps),
        "ROUGE": calculate_rouge(valid_refs, valid_hyps),
        "BERTSCORE": calculate_bertscore(valid_refs, valid_hyps, device),
        "ACC": calculate_acc(valid_refs, valid_hyps),
    }


def run_statistical_analysis(all_seeds_results):
    metrics = list(all_seeds_results[0]["metrics"].keys())
    statistical_results = {}

    for metric in metrics:
        metric_values = [result["metrics"][metric] for result in all_seeds_results]

        if isinstance(metric_values[0], dict):
            statistical_results[metric] = {}
            for sub_metric in metric_values[0].keys():
                if isinstance(metric_values[0][sub_metric], list):
                    mean = np.mean(metric_values[0][sub_metric]) * 100
                    values = [np.mean(result["metrics"][metric][sub_metric]) * 100 for result in all_seeds_results]
                else:
                    mean = np.mean(metric_values[0][sub_metric]) * 100
                    values = [np.mean(result["metrics"][metric][sub_metric]) * 100 for result in all_seeds_results]

                mean = np.mean(values)
                std = np.std(values, ddof=1)
                confidence = 0.95
                degrees_of_freedom = len(values) - 1
                t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
                margin_of_error = t_value * (std / np.sqrt(len(values)))
                conf_interval = (mean - margin_of_error, mean + margin_of_error)
                statistical_results[metric][sub_metric] = {
                    "mean": mean,
                    "std": std,
                    "conf_interval": conf_interval,
                }
        else:
            values = [np.mean(result["metrics"][metric]) * 100 for result in all_seeds_results]
            mean = np.mean(values)
            std = np.std(values, ddof=1)

            confidence = 0.95
            degrees_of_freedom = len(values) - 1
            t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
            margin_of_error = t_value * (std / np.sqrt(len(values)))

            conf_interval = (mean - margin_of_error, mean + margin_of_error)

            statistical_results[metric] = {
                "mean": mean,
                "std": std,
                "conf_interval": conf_interval,
            }

    return statistical_results


def evaluate(elm, dataloader, args):
    show_progress = is_main()
    elm.eval()
    progress = tqdm(
        dataloader,
        desc="Evaluating ELM",
        disable=not show_progress,
        leave=False,
    )
    dataset = dataloader.dataset
    device = next(elm.parameters()).device
    all_refs, all_hyps = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            B = batch["elm_input_ids"].shape[0]
            for b in range(B):
                full_ids = batch["elm_input_ids"][b].tolist()
                full_attn = batch["elm_attention_mask"][b].tolist()
                if args.encoder:
                    full_encoder_tokenizer_out = {
                        k: (v[b].unsqueeze(0) if isinstance(v[b], torch.Tensor) and v[b].ndim > 0 else v[b])
                        for k, v in batch.items()
                        if k.startswith("encoder_") or k in {"ecg_signal", "signal_id_indices", "truncated_padded_ecg_tokens"}
                    }
                ranges = dataset.get_response_ranges(full_ids)
                gt_texts = dataset.get_ground_truth_responses(full_ids, ranges)
                if getattr(args, "dev", False):
                    print(f"\n--- Batch {batch_idx}, Sample {b} ---")
                    print(f"Total turns: {len(ranges)}")
                    dataset.assert_range_alignment(full_ids, ranges)
                for turn_idx, ((s, _), gt) in enumerate(zip(ranges, gt_texts)):
                    sub_ids = full_ids[:s]
                    sub_attn = full_attn[:s]
                    gen_batch = {
                        "elm_input_ids": torch.tensor(sub_ids, dtype=torch.int64).unsqueeze(0),
                        "elm_attention_mask": torch.tensor(sub_attn, dtype=torch.float32).unsqueeze(0),
                    }
                    if args.encoder:
                        gen_batch.update(full_encoder_tokenizer_out)
                    gen_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gen_batch.items()}
                    gen_out = elm.generate(gen_batch)[0].cpu().tolist()
                    gen_txt = dataset.get_generated_response_for_turn(sub_ids, gen_out)
                    if getattr(args, "dev", False):
                        print(f"\nTurn {turn_idx + 1}:")
                        print(f"\nGround Truth:\n{gt}")
                        print(f"\nGenerated:\n{gen_txt}")
                        print("-" * 100)
                    if gt and gen_txt:
                        all_refs.append(gt)
                        all_hyps.append(gen_txt)
            if train_dev_break(getattr(args, "dev", False), batch, 0):
                break
            if batch_idx == 20000:
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
    return {
        "num_pairs": len(all_refs),
        "metrics": results,
        "references": all_refs,
        "hypotheses": all_hyps,
    }
