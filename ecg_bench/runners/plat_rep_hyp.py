### Modified from https://github.com/minyoungg/platonic-rep/blob/main/extract_features.py

import torch
from tqdm import tqdm
import torch.nn.functional as F
import time
import numpy as np
from ecg_bench.utils.plat_rep_metrics import AlignmentMetrics

from ecg_bench.utils.gpu_setup import is_main


def run_plat_rep_hyp_comb(elm, dataloader, args):
    show_progress = is_main()
    elm.eval()
    progress = tqdm(
        dataloader,
        desc="Running Platonic Representation Hypothesis Analysis",
        disable=not show_progress,
        leave=False,
    )
    device = next(elm.parameters()).device

    text_feats = []
    signal_feats = []

    for step, batch in enumerate(progress):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = elm(batch)
        feats = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3)
        feats_mean = feats.mean(dim=1).detach().cpu().float()
        signal_idx = batch["signal_id_indices"][0, 0].detach().cpu()

        signal_feat = feats_mean[:, signal_idx : signal_idx + 1, :].squeeze(1)
        mask = torch.ones(feats_mean.size(1), dtype=torch.bool)
        mask[signal_idx] = False
        text_feat = feats_mean[:, mask, :]
        text_feat = text_feat.mean(dim=1)

        text_feats.append(text_feat)
        signal_feats.append(signal_feat)

        if args.dev and is_main():
            assert torch.allclose(signal_feat.squeeze(1), feats_mean[:, signal_idx, :], atol=1e-6), (
                f"Mismatch at signal_idx={signal_idx}: {signal_feat.squeeze(1)} vs {feats_mean[:, signal_idx, :]}"
            )

        if step > 10 and args.dev and is_main():
            break
        elif step > 5000:
            break

    text_feats = torch.cat(text_feats, dim=0)
    signal_feats = torch.cat(signal_feats, dim=0)
    normalized_text_feats = F.normalize(text_feats, dim=-1)
    normalized_signal_feats = F.normalize(signal_feats, dim=-1)

    trials = 10
    t0 = time.time()
    for metric in AlignmentMetrics.SUPPORTED_METRICS:
        scores, times = [], []
        for t in range(trials):
            t_st = time.time()

            kwargs = {}
            if "nn" in metric:
                kwargs["topk"] = 10
            if "cca" in metric:
                kwargs["cca_dim"] = 10
            if "kernel" in metric:
                kwargs["dist"] = "sample"

            score = AlignmentMetrics.measure(metric, normalized_text_feats, normalized_signal_feats, **kwargs)
            scores.append(score)
            times.append(time.time() - t_st)
        print(f"{metric.rjust(20)}: {np.mean(scores):1.3f} [elapsed: {np.mean(times):.2f}s]")
    print(f"Total time: {time.time() - t0:.2f}s")


def run_plat_rep_hyp_sep(elm, dataloader, args):
    show_progress = is_main()
    elm.eval()
    progress = tqdm(
        dataloader,
        desc="Running Platonic Representation Hypothesis Analysis",
        disable=not show_progress,
        leave=False,
    )
    device = next(elm.parameters()).device

    text_feats = []
    signal_feats = []

    for step, batch in enumerate(progress):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        signal_feat = elm.encoder(batch)
        text_feat = elm.llm.get_llm_embeddings(batch["elm_input_ids"])
        text_feat = text_feat.mean(dim=1)
        text_feats.append(text_feat.detach().cpu().float())
        signal_feats.append(signal_feat.detach().cpu().float())
        if step > 10 and args.dev and is_main():
            break
        elif step > 5000:
            break

    text_feats = torch.cat(text_feats, dim=0)
    signal_feats = torch.cat(signal_feats, dim=0)
    normalized_text_feats = F.normalize(text_feats, dim=-1)
    normalized_signal_feats = F.normalize(signal_feats, dim=-1)

    trials = 10
    t0 = time.time()
    for metric in AlignmentMetrics.SUPPORTED_METRICS:
        scores, times = [], []
        for t in range(trials):
            t_st = time.time()

            kwargs = {}
            if "nn" in metric:
                kwargs["topk"] = 10
            if "cca" in metric:
                kwargs["cca_dim"] = 10
            if "kernel" in metric:
                kwargs["dist"] = "sample"

            score = AlignmentMetrics.measure(metric, normalized_text_feats, normalized_signal_feats, **kwargs)
            scores.append(score)
            times.append(time.time() - t_st)
        print(f"{metric.rjust(20)}: {np.mean(scores):1.3f} [elapsed: {np.mean(times):.2f}s]")
    print(f"Total time: {time.time() - t0:.2f}s")
