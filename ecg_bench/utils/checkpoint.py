import torch
import os
from ecg_bench.utils.gpu_setup import is_main


class CheckpointManager:
    def __init__(self, run_dir, args):
        self.run_dir = run_dir
        self.args = args
        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        self.best_loss = float("inf")
        self.epoch_losses = []
        if is_main():
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, step, is_best=False, prefix=""):
        if not is_main():
            return
        filename = f"{prefix}epoch_{epoch}_step_{step}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        # Handle DDP-wrapped models
        if self.args.distributed:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{prefix}best.pt")
            torch.save(checkpoint, best_path)

    def save_epoch(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.epoch_losses.append(loss)
            return True
        self.epoch_losses.append(loss)
        return False

    def save_step(self, step, total_steps_per_epoch):
        if step == 0:
            return True
        save_interval = max(1, total_steps_per_epoch // 5)
        return step % save_interval == 0

    def stop_early(self):
        if len(self.epoch_losses) < self.args.patience + 1:
            return False
        best_loss = min(self.epoch_losses[: -self.args.patience])
        current_loss = min(self.epoch_losses[-self.args.patience :])
        return current_loss > best_loss - self.args.patience_delta
