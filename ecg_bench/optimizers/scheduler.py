import numpy as np
import argparse
from torch.optim import Adam, AdamW

from ecg_bench.utils.gpu_setup import get_world_size, is_main


def get_optimizer(args, model):
    optimizers = {
        "adam": Adam,
        "adamw": AdamW,
    }
    if args.optimizer.lower() not in optimizers:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported: {list(optimizers.keys())}")
    optimizer_class = optimizers[args.optimizer.lower()]
    optimizer = ScheduledOptim(
        optimizer_class(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            lr=args.lr,
            weight_decay=args.weight_decay,
        ),
        args,
    )
    return optimizer


class ScheduledOptim:
    @staticmethod
    def world_size(args):
        ws = get_world_size()
        if ws == 1 and args.distributed and args.gpus is not None:
            return len(str(args.gpus).split(","))
        return ws

    @staticmethod
    def effective_global_bs(args):
        ws = ScheduledOptim.world_size(args)
        return args.batch_size * int(ws) * args.grad_accum_steps

    def __init__(
        self,
        optimizer,
        args: argparse.Namespace,
    ):
        self.optimizer = optimizer
        self.n_warmup_steps = args.warmup
        self.n_current_steps = 0
        eff_bs = self.effective_global_bs(args)

        if args.ref_global_bs is None:
            args.ref_global_bs = args.batch_size * args.grad_accum_steps
        args.ref_global_bs = max(args.ref_global_bs, 1)
        scale = max(eff_bs / args.ref_global_bs, 1e-8)

        if args.encoder and not args.llm:
            peak_lr = float(getattr(args, "lr", 1e-3)) * scale
            self.init_lr = peak_lr * (self.n_warmup_steps**0.5 if self.n_warmup_steps > 0 else 1.0)
        else:
            peak_lr = float(getattr(args, "lr", 3e-4)) * scale
            self.init_lr = peak_lr * (self.n_warmup_steps**0.5 if self.n_warmup_steps > 0 else 1.0)

        wd_scale = 1.0 if args.scale_wd == "none" else (1.0 / (scale**0.5) if args.scale_wd == "inv_sqrt" else 1.0 / scale)

        for g in self.optimizer.param_groups:
            if "weight_decay" in g and g["weight_decay"] is not None:
                g["weight_decay"] = float(g["weight_decay"]) * wd_scale

        if is_main():
            print(
                f"[scale] eff_bs={eff_bs}, ref_global_bs={args.ref_global_bs}, scale={scale:.4g}, wd_mode={args.scale_wd}, init_lr={self.init_lr:.3e}"
            )

    def step_and_update_lr(self):
        self.update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        step = max(1, self.n_current_steps)
        d_step = 1.0 / np.sqrt(step)
        return min(d_step, step * (self.n_warmup_steps**-1.5)) if self.n_warmup_steps > 0 else d_step

    def update_learning_rate(self):
        lr = max(self.init_lr * self.get_lr_scale(), 1e-8)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.n_current_steps += 1

    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
