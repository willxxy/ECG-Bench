import torch
from tqdm import tqdm


def interpret(model, dataloader, args):
    model.eval()
    show_progress = True
    len_of_batch = 0
    dev_count = 0

    progress_bar = tqdm(dataloader, desc=f"Interpreting {args.model}", disable=not show_progress)

    for step, batch in enumerate(progress_bar):
        if batch is None:
            if show_progress:
                print(f"Skipping invalid batch at step {step}")
            continue

        outputs = model(batch)
        attention = torch.stack(outputs.attentions).mean(dim=(0, 2))[0]
        len_of_batch += 1

        if args.dev:
            dev_count += 1
            if dev_count == 10:
                break
