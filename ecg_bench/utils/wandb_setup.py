import wandb


def setup_wandb(args):
    print("Initializing Wandb")
    wandb.init(
        project="ecg-bench-v2",
        config=args,
    )


def cleanup_wandb():
    wandb.finish()
