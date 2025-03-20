import torch
torch.set_num_threads(2)
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

from ecg_bench.utils.optim_utils import ScheduledOptim
from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.viz_utils import VizUtil
from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer
from ecg_bench.utils.data_loader_utils import FirstStageECGDataset, End2EndECGChatDataset, SecondStageECGChatDataset
from ecg_bench.utils.training_utils import TrainingUtils
from ecg_bench.runners.train import trainer
from ecg_bench.runners.inference import tester, tester_chat
from ecg_bench.runners.post_train import post_trainer_dpo
from ecg_bench.utils.post_train_utils import DPO

def get_args():
    parser = argparse.ArgumentParser(description = None)
    
    ### Data
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data', type=str, default=None, help='Dataset name')
    data_group.add_argument('--seg_len', type=int, default=None, help='Segment length')
    data_group.add_argument('--num_merges', type=int, default=3500, help='Vocab size')
    data_group.add_argument('--target_sf', type=int, default=250, help='Target sampling frequency')
    data_group.add_argument('--pad_to_max', type=int, default=1020, help='Pad to max size')
    data_group.add_argument('--ecg_tokenizer', type=str, help='Tokenizer specification')
    data_group.add_argument('--percentiles', type=str, default=None, help='Percentiles computed during preprocessing')
    data_group.add_argument('--system_prompt', type=str, default=None, help='System prompt')
    data_group.add_argument('--image', action = 'store_true', default=None, help='Turn Image Generation on')
    data_group.add_argument('--instance_normalize', action = 'store_true', default=None, help='Instance normalize ECGs')
    
    ### Model
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model', type=str, default=None, help='Model name')
    model_group.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    model_group.add_argument('--seed', type=int, default=0, help='Random seed')
    
    ### Optimizer
    optim_group = parser.add_argument_group('Optimizer')
    optim_group.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    optim_group.add_argument('--batch_size', type=int, default=128, help='Batch size')
    optim_group.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    optim_group.add_argument('--beta1', type=float, default=0.9, help='Beta1 for optimizer')
    optim_group.add_argument('--beta2', type=float, default=0.99, help='Beta2 for optimizer')
    optim_group.add_argument('--eps', type=float, default=1e-8, help='Epsilon for optimizer')
    optim_group.add_argument('--warmup', type=int, default=500, help='Warmup steps')
    optim_group.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    optim_group.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    optim_group.add_argument('--delta', type=float, default=0.1, help='Delta for early stopping')
    optim_group.add_argument('--attn_implementation', type=str, default=None, help='Attention implementation')
    optim_group.add_argument('--dpo_beta', type=float, default=0.5, help='DPO beta')
    
    ### PEFT
    peft_group = parser.add_argument_group('PEFT')
    peft_group.add_argument('--peft', action='store_true', default=None, help='Use PEFT')
    peft_group.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    peft_group.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    peft_group.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    
    ### Mode and Environment
    mode_group = parser.add_argument_group('Mode and Environment')
    mode_group.add_argument('--train', type=str, choices=['first', 'second', 'end2end'], help='Training mode')
    mode_group.add_argument('--inference', type=str, choices=['second', 'end2end'], help='Inference mode')
    mode_group.add_argument('--post_train', action='store_true', default=None, help='Post-training mode')
    mode_group.add_argument('--interpret', action='store_true', default=None, help='Interpret mode')
    mode_group.add_argument('--dev', action='store_true', default=None, help='Development mode')
    mode_group.add_argument('--log', action='store_true', default=None, help='Enable logging')
    
    ### Distributed Training
    dist_group = parser.add_argument_group('Distributed Training')
    dist_group.add_argument('--dis', action='store_true', default=None, help='Enable distributed training')
    dist_group.add_argument('--gpus', type=str, default='0', help='Comma-separated GPU ids')
    dist_group.add_argument('--ports', type=str, default='12356', help='Comma-separated ports')
    
    ### Checkpoints and Paths
    ckpt_group = parser.add_argument_group('Checkpoints')
    ckpt_group.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    ckpt_group.add_argument('--encoder_checkpoint', type=str, default=None, help='Encoder checkpoint path')

    return parser.parse_args()

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
        args.epochs = 2
        args.log = False
        # args.batch_size = 1  # Changed from 2 to 1
    
    return FileManager(), VizUtil()

def setup_wandb(args):
    """Initialize Weights & Biases logging if enabled"""
    if args.log:
        print('Initializing Wandb')
        wandb.init(
            project='ecg-bench',
            name=f"{'_'.join(args.save_path.split('/')[2:])}",
            config=args
        )

def create_save_path(args):
    base_dir = "./runs"
    dataset_config = f"{args.data}"
    seed_dir = str(args.seed)
    model_params = [
        args.model,
        args.batch_size,
        args.epochs,
        args.lr,
        args.beta1,
        args.beta2,
        args.eps,
        args.warmup,
        args.weight_decay,
        args.instance_normalize
    ]
    model_config = '_'.join(str(param) for param in model_params)    
    save_path = os.path.join(base_dir, dataset_config, seed_dir, model_config)
    return save_path

def save_config(args):    
    args_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    with open(f"{args.save_path}/config.yaml", 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

def save_checkpoint(model, epoch, args, is_best=False):
    model_state_dict = model.module.state_dict() if args.dis else model.state_dict()
    checkpoint = {
        'model': model_state_dict,
        'epoch': epoch
    }
    
    checkpoint_path = f"{args.save_path}/{'best_model' if is_best else f'epoch_{epoch}'}.pth"
    
    if args.dis:
        dist.barrier()
        if dist.get_rank() == 0:
            torch.save(checkpoint, checkpoint_path)
    else:
        torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        print(f"Best model saved at epoch: {epoch+1}")

def run_train(model, train_loader, optimizer, args, viz):
    all_epochs = []
    train_losses = []
    
    for epoch in range(args.epochs):
        all_epochs.append(epoch)
        train_dic = trainer(model, train_loader, optimizer, args, epoch)
        train_losses.append(train_dic['average_loss'])
        
        if args.log:
            wandb.log({
                'train_epoch_loss': train_dic['average_loss'],
                'epoch': epoch
            })
        
        if train_dic['average_loss'] <= min(train_losses):
            save_checkpoint(model, epoch, args, is_best=True)
    
    viz.plot_train_val_loss(train_losses, dir_path=args.save_path)
    
    
def run_post_train(model, test_loader, tokenizer, args, optimizer, judger, dpo, ref_model, viz):
    all_epochs = []
    train_losses = []
    
    checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    print('Model loaded')
    
    ref_checkpoint = torch.load(f"{args.checkpoint}/best_model.pth", map_location=args.device)
    ref_model.load_state_dict(ref_checkpoint['model'])
    print('Reference model loaded')
    
    for epoch in range(args.epochs):
        all_epochs.append(epoch)
        train_dic = post_trainer_dpo(model, test_loader, tokenizer, args, optimizer, epoch, judger, dpo, ref_model)
        train_losses.append(train_dic['average_loss'])
        
        if args.log:
            wandb.log({
                'train_epoch_loss': train_dic['average_loss'],
                'epoch': epoch
            })
        
        if train_dic['average_loss'] <= min(train_losses):
            save_checkpoint(model, epoch, args, is_best=True)
    
    viz.plot_train_val_loss(train_losses, dir_path=args.save_path)

def run_inference(model, test_loader, tokenizer, args, train_utils):
    print(f'Inferencing on {args.model} for checkpoint {args.checkpoint}')
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
        
        with open(f"{args.checkpoint}/seed_{seed}.json", 'w') as f:
            json.dump({
                'averages': seed_results['metrics'],
                'qa_results': seed_results['qa_results']
            }, f)
    
    print(f'Running statistical analysis')
    statistical_results = train_utils.run_statistical_analysis(all_seed_results)
    print(f'Statistical results: {statistical_results}')
    
    with open(f"{args.checkpoint}/statistical_results.json", 'w') as f:
        json.dump(statistical_results, f)

def main(rank, world_size):
    args = get_args()
    device = setup_environment(rank, world_size, args)
    fm, viz = initialize_system(args)
    
    ecg_tokenizer_utils = ECGByteTokenizer(args, fm)
    train_utils = TrainingUtils(args, fm, viz, device, ecg_tokenizer_utils)
    
    args.save_path = create_save_path(args)
    fm.ensure_directory_exists(folder=args.save_path)
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
            model_object['model_hidden_size'], args.warmup)
        
        json_data_file = fm.open_json(f'./data/{args.data}.json')
        if args.inference:
            _, test_data = train_utils.split_dataset(json_data_file)
            data = test_data
            print('Length of Test Dataset:', len(test_data))
        else:
            data = json_data_file
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

        if args.train:
            if args.dis:
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.seed, shuffle=True)
            else:
                sampler = None
            
            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=(sampler is None),
                num_workers=len(args.gpus.split(',')),
                sampler=sampler,
                pin_memory=True)
            
            run_train(model, data_loader, optimizer, args, viz)
        
        elif args.inference:
            data_loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True)
            
            if args.post_train:
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