import torch
torch.set_num_threads(6)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
import torch.multiprocessing as mp
import os
import argparse
from huggingface_hub import login
import gc
import random
import numpy as np
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
import yaml
import copy
import llm_blender

from ecg_bench.config import get_args
from ecg_bench.utils.optim_utils import ScheduledOptim
from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.viz_utils import VizUtil
from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer
from ecg_bench.utils.data_loader_utils import FirstStageECGDataset, End2EndECGChatDataset, SecondStageECGChatDataset
from ecg_bench.utils.training_utils import TrainingUtils
from ecg_bench.runners.train import trainer
from ecg_bench.runners.inference import tester_chat
from ecg_bench.runners.post_train import post_trainer_dpo
from ecg_bench.utils.post_train_utils import DPO

def setup_environment(rank, world_size, args):
    if args.dis:
        print('Setting up Distributed Devices')
        gpu_ids = [int(id) for id in args.gpus.split(',')]
        local_rank = gpu_ids[rank]
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.ports
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        print('Setting up Single Device')
        device = torch.device(args.device)
    
    args.device = device
    return device

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def initialize_system(args):
    print('Loading API key')
    with open('./../.huggingface/api_keys.txt', 'r') as file:
        api_key = file.readlines()[0].strip()
    login(token=api_key)
    
    print('System Cleanup')
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f'Setting Seed to {args.seed}')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.dev:
        print('Running in Development Mode')
        args.epochs = 1
        args.log = False
        # args.batch_size = 1  # Changed from 2 to 1
    
    return FileManager(), VizUtil()

def setup_wandb(args):
    """Initialize Weights & Biases logging if enabled"""
    print('Initializing Wandb')
    wandb.init(
        project='ecg-bench',
        name=f"{'_'.join(args.save_path.split('/')[2:])}",
        config=args
    )

def create_save_path(args, fm):
    if args.train != None or args.post_train != None:
        base_dir = "./runs"
        dataset_config = f"{args.data}"
        seed_dir = str(args.seed)
        if args.encoder_checkpoint != None:
            encoder_in = True
        else:
            encoder_in = False
        model_params = [
            args.model,
            args.batch_size,
            args.epochs,
            args.pad_to_max,
            args.lr,
            args.beta1,
            args.beta2,
            args.eps,
            args.warmup,
            args.weight_decay,
            args.instance_normalize,
            args.image,
            args.augment_image,
            args.train_encoder,
            args.rag
        ]
        
        if args.rag:
            model_params.extend([
                args.retrieval_base,
                args.retrieved_information,
                args.rag_k,
                args.rag_prompt_mode,
                args.normalized_rag_feature
            ])
            
        model_params.append(encoder_in)
        model_config = '_'.join(str(param) for param in model_params)
        save_path = os.path.join(base_dir, dataset_config, seed_dir, model_config)
        fm.ensure_directory_exists(folder=save_path)
        return save_path
    else:
        return args.checkpoint

def save_config(args):    
    args_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    with open(f"{args.save_path}/config.yaml", 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

def save_checkpoint(model, epoch, args, is_best=False):
    # Only save the model on the main process to avoid corruption from race conditions
    if args.dis:
        # Add barrier to synchronize all processes before saving
        dist.barrier()
        if dist.get_rank() == 0:
            model_state_dict = model.module.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'epoch': epoch
            }
            checkpoint_path = f"{args.save_path}/{'best_model' if is_best else f'epoch_{epoch}'}.pth"
            torch.save(checkpoint, checkpoint_path)
            # Print only from main process
            if is_best:
                print(f"Best model saved at epoch: {epoch+1}")
        # Add another barrier to ensure all processes wait until the save is complete
        dist.barrier()
    else:
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'epoch': epoch
        }
        checkpoint_path = f"{args.save_path}/{'best_model' if is_best else f'epoch_{epoch}'}.pth"
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            print(f"Best model saved at epoch: {epoch+1}")

def run_train(model, train_loader, optimizer, args, viz):
    all_epochs = []
    train_losses = []
    best_loss = float('inf')
    
    if args.checkpoint != None:
        checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded and training resumed')
    
    for epoch in range(args.epochs):
        all_epochs.append(epoch)
        train_dic = trainer(model, train_loader, optimizer, args, epoch)
        current_loss = train_dic['average_loss']
        train_losses.append(current_loss)
        
        if args.log:
            wandb.log({
                'train_epoch_loss': current_loss,
                'epoch': epoch
            })
        
        # Only save model if the loss is finite and better than previous best
        if current_loss < best_loss and current_loss != float('inf'):
            best_loss = current_loss
            save_checkpoint(model, epoch, args, is_best=True)
            if args.dis and dist.get_rank() == 0 or not args.dis:
                print(f"New best loss: {best_loss} at epoch {epoch+1}")
    
    # Only use finite losses for plotting
    finite_losses = [loss for loss in train_losses if loss != float('inf')]
    if finite_losses:
        viz.plot_train_val_loss(finite_losses, dir_path=args.save_path)
    else:
        print("Warning: No finite losses recorded during training")

def run_post_train(model, test_loader, tokenizer, args, optimizer, judger, dpo, ref_model, viz):
    all_epochs = []
    train_losses = []
    best_loss = float('inf')
    
    checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    print('Model loaded')
    
    ref_checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=args.device)
    ref_model.load_state_dict(ref_checkpoint['model'])
    print('Reference model loaded')
    
    for epoch in range(args.epochs):
        all_epochs.append(epoch)
        train_dic = post_trainer_dpo(model, test_loader, tokenizer, args, optimizer, epoch, judger, dpo, ref_model)
        current_loss = train_dic['average_loss']
        train_losses.append(current_loss)
        
        if args.log:
            wandb.log({
                'train_epoch_loss': current_loss,
                'epoch': epoch
            })
        
        # Only save model if the loss is finite and better than previous best
        if current_loss < best_loss and current_loss != float('inf'):
            best_loss = current_loss
            save_checkpoint(model, epoch, args, is_best=True)
            if args.dis and dist.get_rank() == 0 or not args.dis:
                print(f"New best loss: {best_loss} at epoch {epoch+1}")
    
    # Only use finite losses for plotting
    finite_losses = [loss for loss in train_losses if loss != float('inf')]
    if finite_losses:
        viz.plot_train_val_loss(finite_losses, dir_path=args.save_path)
    else:
        print("Warning: No finite losses recorded during training")

def run_inference(model, test_loader, tokenizer, args, train_utils):
    print(f'Inferencing on {args.model} for checkpoint {args.checkpoint}')
    # seeds = [0, 1, 2, 3, 4]
    seeds = [0, 1]
    all_seed_results = []
    
    for seed in seeds:
        print(f'Inferencing on seed {seed}')
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')
        
        seed_results = tester_chat(model, test_loader, tokenizer, args, train_utils)
        all_seed_results.append(seed_results)
        
        # Construct filename based on args.rag
        if args.rag:
            filename = f"seed_{seed}_{args.perturb}_{args.rag}_{args.retrieval_base}_{args.retrieved_information}_{args.rag_k}_{args.rag_prompt_mode}_{args.normalized_rag_feature}.json"
        else:
            filename = f"seed_{seed}_{args.perturb}_{args.rag}.json"
            
        with open(f"{args.checkpoint}/{filename}", 'w') as f:
            json.dump({
                'averages': seed_results['metrics'],
                'qa_results': seed_results['qa_results']
            }, f)
    
    print(f'Running statistical analysis')
    statistical_results = train_utils.run_statistical_analysis(all_seed_results)
    print(f'Statistical results: {statistical_results}')
    
    # Update statistical results filename similarly
    if args.rag:
        stat_filename = f"statistical_results_{args.perturb}_{args.rag}_{args.retrieval_base}_{args.retrieved_information}_{args.rag_k}_{args.rag_prompt_mode}_{args.normalized_rag_feature}.json"
    else:
        stat_filename = f"statistical_results_{args.perturb}_{args.rag}.json"
        
    with open(f"{args.checkpoint}/{stat_filename}", 'w') as f:
        json.dump(statistical_results, f)

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {
            'input_ids': torch.tensor([], dtype=torch.int64),
            'attn_mask': torch.tensor([], dtype=torch.float32),
            'assistant_ranges': []
        }
    return torch.utils.data.dataloader.default_collate(batch)

def main(rank, world_size):
    args = get_args()
    device = setup_environment(rank, world_size, args)
    fm, viz = initialize_system(args)
    
    ecg_tokenizer_utils = ECGByteTokenizer(args, fm)
    train_utils = TrainingUtils(args, fm, viz, device, ecg_tokenizer_utils)
    
    args.save_path = create_save_path(args, fm)
    if args.log:
        setup_wandb(args)
    save_config(args)
    
    try:
        print(f'Creating Model: {args.model}')
        model_object = train_utils.create_model()
        
        if args.train == 'first':
            model = model_object['encoder']
            tokenizer = model_object['encoder_tokenizer']
        elif args.train == 'second' or args.inference == 'second':
            model = model_object['llava']
            tokenizer = model_object['llm_tokenizer']
        elif args.train == 'end2end' or args.inference == 'end2end':
            model = model_object['llm']
            tokenizer = model_object['llm_tokenizer']
        
        if args.dis:
            model = DDP(model, device_ids=[device.index], find_unused_parameters=model_object['find_unused_parameters'])
        
        print(f'Total number of parameters: {train_utils.count_parameters(model)}')
        
        optimizer = ScheduledOptim(
            Adam(filter(lambda x: x.requires_grad, model.parameters()),
                 betas=(args.beta1, args.beta2), eps=args.eps, lr=args.lr, weight_decay=args.weight_decay),
            model_object['model_hidden_size'], args)
        
        json_data_file = fm.open_json(f'./data/{args.data}.json')
        
        train_data, test_data = train_utils.split_dataset(json_data_file)
        if args.train == 'first':
            data = train_data[:800000]
        elif args.train in ['second', 'end2end']:
            data = train_data[:400000]
        elif args.inference in ['second', 'end2end']:
            data = test_data[:20000]
        print('Length of Dataset:', len(data))
        
        if args.train == 'first':
            dataset = FirstStageECGDataset(
                json_data_file=data,
                train_utils=train_utils,
                encoder_tokenizer=model_object.get('encoder_tokenizer'))
        elif args.train == 'second' or args.inference == 'second':
            dataset = SecondStageECGChatDataset(
                json_data_file=data,
                train_utils=train_utils,
                llm_tokenizer=tokenizer,
                encoder_tokenizer=model_object.get('encoder_tokenizer'))
        elif args.train == 'end2end' or args.inference == 'end2end':
            dataset = End2EndECGChatDataset(
                json_data_file=data,
                train_utils=train_utils,
                llm_tokenizer=tokenizer)

        if args.train != None:
            if args.dis:
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.seed, shuffle=True)
            else:
                sampler = None
            
            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=(sampler is None),
                num_workers=0,
                sampler=sampler,
                pin_memory=True,
                collate_fn=collate_fn)
            
            run_train(model, data_loader, optimizer, args, viz)
        
        elif args.inference != None:
            data_loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                collate_fn=collate_fn)
            
            if args.post_train != None:
                ### FROM LLM-BLENDER
                judger = llm_blender.Blender()
                judger.loadranker("llm-blender/PairRM", device = device, cache_dir = './../.huggingface')
                dpo = DPO(beta = args.dpo_beta)
                ref_model = copy.deepcopy(model)
                run_post_train(model, data_loader, tokenizer, args, optimizer, judger, dpo, ref_model, viz)
            else:
                run_inference(model, data_loader, tokenizer, args, train_utils)
        
    finally:
        if args.log:
            wandb.finish()
        if args.dis:
            cleanup()

if __name__ == '__main__':
    args = get_args()
    world_size = len(args.gpus.split(','))
    if args.dis:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        rank = 0
        world_size = 1
        main(rank, world_size) 