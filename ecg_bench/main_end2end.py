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

from ecg_bench.utils.optim_utils import ScheduledOptim
from ecg_bench.utils.dir_file_utils import FileManager
from ecg_bench.utils.viz_utils import VizUtil
from ecg_bench.utils.ecg_tokenizer_utils import ECGByteTokenizer
from ecg_bench.utils.data_loader_utils import ECGDataset
from ecg_bench.utils.training_utils import TrainingUtils
from ecg_bench.runners.train import trainer
from ecg_bench.runners.inference import tester

def get_args():
    parser = argparse.ArgumentParser(description = None)
    
    ### Data
    parser.add_argument('--data', type = str, default = None, help = 'Please choose the dataset')
    parser.add_argument('--seg_len', type = int, default = 500, help = 'Please choose the segment length')
    parser.add_argument('--num_merges', type = int, default = 3500, help = 'Please specify the vocab size')
    parser.add_argument('--target_sf', type = int, default = 250, help = 'Please choose the target sampling frequency')
                        
    ### Model
    parser.add_argument('--model', type = str, default = None, help='Please choose the model')
    parser.add_argument('--device', type = str, default = None, help='Please choose the device')
    parser.add_argument('--seed', type = int, default = 0, help='Please choose the seed')
    parser.add_argument('--pad_to_max', type = int, default = 1020, help = 'Please specify the pad to max size')
    parser.add_argument('--ecg_tokenizer', type = str, help = 'Please specify the tokenizer')
    parser.add_argument('--percentiles', type = str, default = None, help = 'Please choose the percentiles computed during preprocessing')
    
    ### Optimizer
    parser.add_argument('--lr', type = float, default = 1e-4, help='Please choose the learning rate')
    parser.add_argument('--batch_size', type = int, default = 128, help='Please choose the batch size')
    parser.add_argument('--epochs', type = int, default = 150, help='Please choose the number of epochs')
    parser.add_argument('--beta1', type = float, default = 0.9, help='Please choose beta 1 for optimizer')
    parser.add_argument('--beta2', type = float, default = 0.99, help='Please choose beta 2 for optimizer')
    parser.add_argument('--eps', type = float, default = 1e-8, help='Please choose epsilon for optimizer')
    parser.add_argument('--warmup', type = int, default = 500, help = 'Please choose the number of warmup steps for the optimizer' )
    parser.add_argument('--weight_decay', type = float, default = 1e-2, help = 'Please choose the weight decay')
    parser.add_argument('--patience', type = int, default = 5, help='Please choose the patience')
    parser.add_argument('--delta', type = float, default = 0.1, help='Please choose the delta')
    
    ### PEFT 
    parser.add_argument('--peft', action = 'store_true', default = None, help = 'Please choose whether to use PEFT or not')
    parser.add_argument('--lora_rank', type = int, default = 16, help = 'Please choose the lora rank')
    parser.add_argument('--lora_alpha', type = int, default = 32, help = 'Please choose the lora alpha')
    parser.add_argument('--lora_dropout', type = float, default = 0.05, help = 'Please choose the lora dropout')
    
    ### For development
    parser.add_argument('--dev', action = 'store_true', default = None, help = 'Please choose whether to use development mode or not')
    parser.add_argument('--log', action = 'store_true', default = None, help = 'Please choose whether to log or not')
    
    ### Distributed Training
    parser.add_argument('--dis', action = 'store_true', default = None, help = 'Please choose whether to distributed training or not')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU ids to use (e.g., "0,1,2")')
    parser.add_argument('--ports', type=str, default='12356', help='Comma-separated list of ports to use (e.g., "12355,12356,12357")')
    
    ### Mode
    parser.add_argument('--train', type = str, default = None, help = 'Please choose the training mode [first, second, end2end]')
    parser.add_argument('--interpret', action = 'store_true', default = None, help = 'Please choose whether to interpret the model or not')
    parser.add_argument('--inference', type = str, default = None, help = 'Please choose the inference mode [second, end2end]')
    
    ### For inference
    parser.add_argument('--checkpoint', type = str, default = None, help = 'Please choose the checkpoint')
    
    return parser.parse_args()

def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.ports
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    args = get_args()
    
    if args.dev:
        print('Running in Development Mode')
        args.epochs=2
        args.log = False
        args.batch_size = 2
    
    if args.dis:
        print('Setting up Distributed Devices')
        gpu_ids = [int(id) for id in args.gpus.split(',')]
        local_rank = gpu_ids[rank]
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        args.device = device
        setup(rank, world_size, args)
    else:
        print('Setting up Single Device')
        device = torch.device(args.device)
        
    print('Loading API key')
    with open('./../.huggingface/api_keys.txt', 'r') as file:
        file_contents = file.readlines()
    api_key = file_contents[0].strip()
    login(token = api_key)
    
    print('Collecting Garbage')
    gc.collect()
    
    print('Emptying CUDA Cache')
    torch.cuda.empty_cache()
    
    print(f'Setting Seed to {args.seed}')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print('Initializing Objects')
    fm = FileManager()
    viz = VizUtil()
    ecg_tokenizer_utils = ECGByteTokenizer(args, fm)
    train_utils = TrainingUtils(args, fm, viz, device, ecg_tokenizer_utils)
    
    print('Creating runs directory')
    args.save_path = f"./runs/{args.data}_{args.seg_len}_{args.num_merges}_{args.target_sf}/{args.seed}/{args.model}_{args.batch_size}_{args.epochs}_{args.lr}_{args.beta1}_{args.beta2}_{args.eps}_{args.warmup}_{args.weight_decay}"
    fm.ensure_directory_exists(folder = args.save_path)
    
    if args.log:
        print('Initializing Wandb')
        wandb.init(project = 'ecg-bench',
                   name = f"{'_'.join(args.save_path.split('/')[2:])}",
                   config = args)
    
    print(f'Creating Model: {args.model}')
    model_object = train_utils.create_model()
    
    llm = model_object['llm']
    llm_tokenizer = model_object['llm_tokenizer']
    llm = llm.to(device)
    if args.dis:
        llm = DDP(llm, device_ids=[local_rank], find_unused_parameters=model_object['find_unused_parameters'])
    
    print(f'Total number of parameters: {train_utils.count_parameters(llm)}')
    
    optimizer = ScheduledOptim(
        Adam(filter(lambda x: x.requires_grad, llm.parameters()),
            betas=(args.beta1, args.beta2), eps=args.eps, lr = args.lr, weight_decay=args.weight_decay), 
                    model_object['model_hidden_size'], args.warmup)
    
    
    json_data_file = fm.open_json(f'./data/{args.data}.json')
    train_data, test_data = train_utils.split_dataset(json_data_file)
    print('Length of Train Dataset:', len(train_data))
    print('Length of Test Dataset:', len(test_data))
    
    if args.train == 'end2end' and args.inference == None:
        train_dataset = ECGDataset(
            json_data_file = train_data,
            args = args,
            train_utils = train_utils,
            llm_tokenizer = llm_tokenizer)
        
        if args.dis:
            train_sampler = DistributedSampler(train_dataset, 
                                            num_replicas=world_size,
                                            rank=rank, 
                                            seed=args.seed,
                                            shuffle=True)
        else:
            train_sampler = None

        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=(train_sampler is None),  # shuffle if no sampler
                                num_workers=len(args.gpus.split(',')),
                                sampler=train_sampler,
                                pin_memory=True)

        all_epochs = []
        train_losses = []
        
        for epoch in range(args.epochs):
            all_epochs.append(epoch)
            train_dic = trainer(llm, train_loader, optimizer, args, epoch)
            train_losses.append(train_dic['average_loss'])
            
            if args.log:
                wandb.log({
                    'train_epoch_loss' : train_dic['average_loss'],
                    'epoch' : epoch
                })
            
            if train_dic['average_loss'] <= min(train_losses):
                model_state_dict = llm.module.state_dict() if args.dis else llm.state_dict()
                
                checkpoint = {
                    'model': model_state_dict,
                    'epoch': epoch
                }
                
                checkpoint_path = f"{args.save_path}/best_model.pth"
                
                if args.dis:
                    dist.barrier()
                    if dist.get_rank() == 0:
                        torch.save(checkpoint, checkpoint_path)
                else:
                    torch.save(checkpoint, checkpoint_path)
                
                print(f"Best model saved at epoch: {epoch+1}")
                
        viz.plot_train_val_loss(train_losses, dir_path = args.save_path)
            
        
        if args.log:
            wandb.finish()
            
        if args.dis:
            cleanup()
            
    elif args.train == None and args.inference == 'end2end':
        test_dataset = ECGDataset(
            json_data_file = test_data,
            args = args,
            train_utils = train_utils,
            llm_tokenizer = llm_tokenizer)
            
        test_loader = DataLoader(test_dataset,
                                batch_size = 1,
                                shuffle = False,
                                pin_memory = True)
        print(f'Inferencing on {args.model} for checkpoint {args.checkpoint}')
        seeds = [0, 1]
        all_seed_results = []
        checkpoint_path = f"./runs/{args.data}_{args.seg_len}_{args.num_merges}_{args.target_sf}/{args.seed}/{args.checkpoint}"
        for seed in seeds:
            print(f'Inferencing on seed {seed}')
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            
            checkpoint = torch.load(f"{checkpoint_path}/best_model.pth", map_location=args.device)
            llm.load_state_dict(checkpoint['model'])
            print('Model loaded')
            
            seed_results = tester(llm, test_loader, llm_tokenizer, args, train_utils)
            all_seed_results.append(seed_results)
            with open(f"{checkpoint_path}/seed_{seed}.json", 'w') as f:
                json.dump({
                    'averages': seed_results['metrics'],
                    'qa_results': seed_results['qa_results']
                }, f)
        
        print(f'Inferencing on {args.model} for checkpoint {args.checkpoint} completed')
        print(f'Running statistical analysis')
        statistical_results = train_utils.run_statistical_analysis(all_seed_results)
        print(f'Statistical results: {statistical_results}')
        
        with open(f"{checkpoint_path}/statistical_results.json", 'w') as f:
            json.dump(statistical_results, f)
            
if __name__ == '__main__':
    args = get_args()
    world_size = len(args.gpus.split(','))
    if args.dis:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        rank = 0
        world_size = 1
        main(rank, world_size)