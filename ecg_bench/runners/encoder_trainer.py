import torch
import wandb
from tqdm import tqdm
from ecg_bench.utils.gpu_setup import is_main, train_dev_break


def train(encoder, dataloader, optimizer, epoch, args, checkpoint_manager=None):
    if getattr(args, "distributed", False) and hasattr(getattr(dataloader, "sampler", None), "set_epoch"):
        dataloader.sampler.set_epoch(epoch)
    show_progress = is_main()
    encoder.train()
    total_loss = 0
    total_steps = 0
    progress = tqdm(
        dataloader,
        desc=f"Training Encoder; Epoch {epoch}",
        disable=not show_progress,
        leave=False,
    )
    device = next(encoder.parameters()).device
    total_steps_per_epoch = len(dataloader)

    for step, batch in enumerate(progress):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = encoder(batch)
        loss = outputs.loss
        total_loss += loss.item()
        total_steps += 1
        loss.backward()
        optimizer.step_and_update_lr()
        if getattr(args, "wandb", False) and is_main():
            wandb.log({"train/step_loss": loss.item(), "epoch": epoch})
        if checkpoint_manager and checkpoint_manager.save_step(step, total_steps_per_epoch):
            checkpoint_manager.save_checkpoint(encoder, optimizer, epoch, step, prefix="step_")
        if train_dev_break(getattr(args, "dev", False), batch, loss.item()):
            break
    average_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    return {"average_loss": average_loss, "total_steps": total_steps}
