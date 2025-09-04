import torch

torch.set_num_threads(6)
import copy
import gc
import json
import os
import random
from operator import itemgetter

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from huggingface_hub import HfFolder, login
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ecg_bench.config import get_args
from ecg_bench.runners.inference import tester_chat
from ecg_bench.runners.post_train import post_trainer_dpo
from ecg_bench.runners.train import trainer
from ecg_bench.utils.data_loader_utils import (
    End2EndECGChatDataset,
    FirstStageECGDataset,
    SecondStageECGChatDataset,
)
from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer
from ecg_bench.utils.optim_utils import ScheduledOptim
from ecg_bench.utils.training_utils import TrainingUtils
from ecg_bench.utils.viz_utils import VizUtil


def setup_environment(rank, world_size, args):
    if args.dis:
        print("Setting up Distributed Devices")
        gpu_ids = [int(id) for id in args.gpus.split(",")]
        local_rank = gpu_ids[rank]
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = args.ports
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        print("Setting up Single Device")
        device = torch.device(args.device)

    args.device = device
    return device

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def initialize_system(args):
    print("Loading API key")
    if HfFolder.get_token() is None:
        print("Loading API key and logging in …")
        with open("./../.huggingface/api_keys.txt") as f:
            api_key = f.readline().strip()
        login(token=api_key)
    else:
        print("Hugging Face token already present skipping login.")

    print("System Cleanup")
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Setting Seed to {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dev:
        print("Running in Development Mode")
        args.epochs = 1
        args.log = False
        args.batch_size = 1

    return FileManager(), VizUtil()

def create_save_path(args, fm):
    if args.train is not None or args.post_train is not None:
        base_dir = "./runs"
        dataset_config = f"{args.data}"
        seed_dir = str(args.seed)
        encoder_in = True if args.encoder_checkpoint is not None else False
        model_params = [
            args.model,
            args.optimizer,
            args.batch_size,
            args.epochs,
            args.pad_to_max,
            args.lr,
            args.beta1,
            args.beta2,
            args.eps,
            args.warmup,
            args.weight_decay,
            args.image,
            args.augment_image,
            args.train_encoder,
            args.rag,
            args.fold,
            args.blackout,
            args.no_signal,
        ]

        if args.rag:
            model_params.extend([
                args.retrieval_base,
                args.retrieved_information,
                args.rag_k,
                args.rag_prompt_mode,
            ])

        model_params.append(encoder_in)
        model_config = "_".join(str(param) for param in model_params)
        save_path = os.path.join(base_dir, dataset_config, seed_dir, model_config)
        fm.ensure_directory_exists(folder=save_path)
        return save_path
    return args.checkpoint


def save_checkpoint(model, epoch, args, is_best=False):
    if args.dis:
        dist.barrier()
        if dist.get_rank() == 0:
            model_state_dict = model.module.state_dict()
            checkpoint = {
                "model": model_state_dict,
                "epoch": epoch,
            }
            checkpoint_path = f"{args.save_path}/{'best_model' if is_best else f'epoch_{epoch}'}.pth"
            torch.save(checkpoint, checkpoint_path)
            if is_best: print(f"Best model saved at epoch: {epoch+1}")
        dist.barrier()
    else:
        model_state_dict = model.state_dict()
        checkpoint = {
            "model": model_state_dict,
            "epoch": epoch,
        }
        checkpoint_path = f"{args.save_path}/{'best_model' if is_best else f'epoch_{epoch}'}.pth"
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            print(f"Best model saved at epoch: {epoch+1}")

def run_train(model, train_loader, optimizer, args, viz):
    all_epochs = []
    train_losses = []
    best_loss = float("inf")

    if args.checkpoint is not None:
        checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        print("Model loaded and training resumed")

    for epoch in range(args.epochs):
        all_epochs.append(epoch)
        train_dic = trainer(model, train_loader, optimizer, args, epoch)
        current_loss = train_dic["average_loss"]
        train_losses.append(current_loss)

        if current_loss < best_loss and current_loss != float("inf"):
            best_loss = current_loss
            save_checkpoint(model, epoch, args, is_best=True)
            if (args.dis and dist.get_rank() == 0) or not args.dis: print(f"New best loss: {best_loss} at epoch {epoch+1}")

    finite_losses = [loss for loss in train_losses if loss != float("inf")]
    if finite_losses: viz.plot_train_val_loss(finite_losses, dir_path=args.save_path)
    else: print("Warning: No finite losses recorded during training")

def run_post_train(model, test_loader, tokenizer, args, optimizer, judger, dpo, ref_model, viz):
    all_epochs = []
    train_losses = []
    best_loss = float("inf")

    checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    print("Model loaded")

    ref_checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=args.device)
    ref_model.load_state_dict(ref_checkpoint["model"])
    print("Reference model loaded")

    for epoch in range(args.epochs):
        all_epochs.append(epoch)
        train_dic = post_trainer_dpo(model, test_loader, tokenizer, args, optimizer, epoch, judger, dpo, ref_model)
        current_loss = train_dic["average_loss"]
        train_losses.append(current_loss)

        # Only save model if the loss is finite and better than previous best
        if current_loss < best_loss and current_loss != float("inf"):
            best_loss = current_loss
            save_checkpoint(model, epoch, args, is_best=True)
            if (args.dis and dist.get_rank() == 0) or not args.dis: print(f"New best loss: {best_loss} at epoch {epoch+1}")

    # Only use finite losses for plotting
    finite_losses = [loss for loss in train_losses if loss != float("inf")]
    if finite_losses: viz.plot_train_val_loss(finite_losses, dir_path=args.save_path)
    else: print("Warning: No finite losses recorded during training")

def run_inference(model, test_loader, tokenizer, args, train_utils):
    print(f"Inferencing on {args.model} for checkpoint {args.checkpoint}")
    seeds = [0, 1]
    all_seed_results = []

    for seed in seeds:
        print(f"Inferencing on seed {seed}")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        print("Model loaded")

        seed_results = tester_chat(model, test_loader, tokenizer, args, train_utils)
        all_seed_results.append(seed_results)

        # Construct filename based on args.rag
        filename = f"seed_{seed}_{args.perturb}_{args.rag}_{args.blackout}_{args.no_signal}.json"

        with open(f"{args.checkpoint}/{filename}", "w") as f:
            json.dump({
                "averages": seed_results["metrics"],
                "qa_results": seed_results["qa_results"],
            }, f)

    print("Running statistical analysis")
    statistical_results = train_utils.run_statistical_analysis(all_seed_results)
    print(f"Statistical results: {statistical_results}")

    # Update statistical results filename similarly
    stat_filename = f"statistical_results_{args.perturb}_{args.rag}_{args.blackout}_{args.no_signal}.json"

    with open(f"{args.checkpoint}/{stat_filename}", "w") as f:
        json.dump(statistical_results, f)

def main(rank, world_size):
    args = get_args()
    device = setup_environment(rank, world_size, args)
    fm, viz = initialize_system(args)

    ecg_tokenizer_utils = ECGByteTokenizer(args, fm)
    train_utils = TrainingUtils(args, fm, viz, device, ecg_tokenizer_utils)

    args.save_path = create_save_path(args, fm)
    if args.log: train_utils.setup_wandb()
    if args.inference == None: train_utils.save_config()

    try:
        print(f"Creating Model: {args.model}")
        model_object = train_utils.create_model()

        if args.train == "first": model, tokenizer = itemgetter("encoder", "encoder_tokenizer")(model_object)
        elif args.train == "second" or args.inference == "second": model, tokenizer = itemgetter("llava", "llm_tokenizer")(model_object)
        elif args.train == "end2end" or args.inference == "end2end": model, tokenizer = itemgetter("llm", "llm_tokenizer")(model_object)

        if args.dis:
            model = DDP(model, device_ids=[device.index],
                        find_unused_parameters=model_object["find_unused_parameters"])

        print(f"Total number of parameters: {train_utils.count_parameters(model)}")

        if args.train:
            optimizer_class = train_utils.get_optimizer_class(args.optimizer)
            optimizer = ScheduledOptim(
                optimizer_class(filter(lambda x: x.requires_grad, model.parameters()),
                    betas=(args.beta1, args.beta2), eps=args.eps, lr=args.lr, weight_decay=args.weight_decay),
                    model_object["model_hidden_size"], args)
            train_data = load_dataset(f"willxxy/{args.data}", split=f"fold{args.fold}_train").with_transform(fm.decode_batch)
            print(f"Length of Train Data: {len(train_data)}")
        elif args.inference:
            test_data = load_dataset(f"willxxy/{args.data}", split=f"fold{args.fold}_test").with_transform(fm.decode_batch)
            print(f"Length of Test Data: {len(test_data)}")

        if args.train == "first": data = train_data.select(range(800000))
        elif args.train in ["second", "end2end"]: data = train_data.select(range(400000))
        elif args.inference in ["second", "end2end"]: data = test_data.select(range(20000))
        print("Length of Dataset Considered:", len(data))

        if args.train == "first":
            dataset = FirstStageECGDataset(
                json_data_file=data,
                train_utils=train_utils,
                encoder_tokenizer=model_object.get("encoder_tokenizer"))
        elif args.train == "second" or args.inference == "second":
            dataset = SecondStageECGChatDataset(
                json_data_file=data,
                train_utils=train_utils,
                llm_tokenizer=tokenizer,
                encoder_tokenizer=model_object.get("encoder_tokenizer"))
        elif args.train == "end2end" or args.inference == "end2end":
            dataset = End2EndECGChatDataset(
                json_data_file=data,
                train_utils=train_utils,
                llm_tokenizer=tokenizer)

        if args.train is not None:
            if args.dis: sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.seed, shuffle=True)
            else: sampler = None

            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=(sampler is None),
                num_workers=2 if args.train == "first" else 0,
                sampler=sampler,
                pin_memory=True,
                collate_fn=train_utils.collate_fn)

            run_train(model, data_loader, optimizer, args, viz)

        elif args.inference is not None:
            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=True,
                collate_fn=train_utils.collate_fn)

            if args.post_train is not None:
                ### FROM LLM-BLENDER
                import llm_blender
                judger = llm_blender.Blender()
                judger.loadranker("llm-blender/PairRM", device = device, cache_dir = train_utils.cache_dir)

                from ecg_bench.utils.post_train_utils import DPO
                dpo = DPO(beta = args.dpo_beta)
                ref_model = copy.deepcopy(model)

                run_post_train(model, data_loader, tokenizer, args, optimizer, judger, dpo, ref_model, viz)
            else: run_inference(model, data_loader, tokenizer, args, train_utils)

    finally:
        if args.dis:
            cleanup()

if __name__ == "__main__":
    args = get_args()
    world_size = len(args.gpus.split(","))
    if args.dis: mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else: main(0, 1)
