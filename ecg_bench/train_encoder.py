import torch
import gc

torch.set_num_threads(6)
from ecg_bench.configs.config import get_args
from ecg_bench.utils.gpu_setup import init_dist, cleanup, GPUSetup, is_main
from ecg_bench.utils.set_seed import set_seed
from ecg_bench.dataloaders.build_dataloader import BuildDataLoader
from ecg_bench.elms.build_encoder import BuildEncoder
from ecg_bench.runners.encoder_trainer import train
from ecg_bench.utils.file_manager import setup_experiment_folders
from ecg_bench.configs.constants import RUNS_DIR
from ecg_bench.optimizers.scheduler import get_optimizer
from ecg_bench.utils.checkpoint import CheckpointManager
from ecg_bench.utils.wandb_setup import setup_wandb, cleanup_wandb
import wandb


def main():
    gc.collect()
    torch.cuda.empty_cache()
    mode = "train"
    args = get_args(mode)

    if args.distributed:
        init_dist()

    run_dir = setup_experiment_folders(
        f"{RUNS_DIR}/training/encoder",
        args,
    )
    if is_main():
        print(f"Run dir: {run_dir}")
        if args.wandb:
            setup_wandb(args)

    set_seed(getattr(args, "seed", 1337))

    build_dataloader = BuildDataLoader(mode, args)
    dataloader = build_dataloader.build_dataloader()

    build_encoder = BuildEncoder(args)
    encoder_components = build_encoder.build_encoder()

    gpu_setup = GPUSetup(args)
    encoder = gpu_setup.setup_gpu(encoder_components["encoder"], encoder_components["find_unused_parameters"])
    if args.dev:
        gpu_setup.print_model_device(encoder, f"{args.encoder}")

    optimizer = get_optimizer(args, encoder)
    checkpoint_manager = CheckpointManager(run_dir, args)

    for epoch in range(args.epochs):
        train_result = train(encoder, dataloader, optimizer, epoch, args, checkpoint_manager)
        if args.wandb and is_main():
            wandb.log({"train/epoch_loss": train_result["average_loss"], "epoch": epoch})

        if checkpoint_manager.save_epoch(train_result["average_loss"]):
            checkpoint_manager.save_checkpoint(encoder, optimizer, epoch, -1, is_best=True, prefix="epoch_")
        if checkpoint_manager.stop_early():
            if is_main():
                print(f"Early stopping at epoch {epoch}")
            break

    if args.distributed:
        cleanup()

    if is_main() and args.wandb:
        cleanup_wandb()


if __name__ == "__main__":
    main()
