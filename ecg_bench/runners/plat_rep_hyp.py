### Modified from https://github.com/minyoungg/platonic-rep/blob/main/extract_features.py

import torch
from tqdm import tqdm

from ecg_bench.utils.gpu_setup import is_main


def run_plat_rep_hyp(elm, dataloader, args):
    show_progress = is_main()
    elm.eval()
    progress = tqdm(
        dataloader,
        desc="Running Platonic Representation Hypothesis Analysis",
        disable=not show_progress,
        leave=False,
    )
    device = next(elm.parameters()).device

    for step, batch in enumerate(progress):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = elm(batch)
        feats = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3)
        print("batch['elm_input_ids'].shape", batch["elm_input_ids"].shape)
        print("feats", feats.shape)
        mask = batch["elm_attention_mask"].unsqueeze(-1).unsqueeze(1)
        feats = (feats * mask).sum(2) / mask.sum(2)
        print("averaged feats", feats.shape)
        print("attention mask", mask.shape)
        break
