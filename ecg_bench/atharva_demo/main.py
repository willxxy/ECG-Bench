'''
1. How do we send both question and answer ?
2. How does one epoch of training look like ?
3. How does one epoch of validation/testing look like ?
4. 
'''

import torch
torch.set_num_threads(4)
import random
import numpy as np
from torch.utils.data import DataLoader
import argparse
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
# import torch.multiprocessing as mp
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import login

from src.dataset import ECGChatDataset
# from scheduler import ScheduledOptim
from src.epoch import *
from src.model import LLM
from src.epoch import *
from src.utils import *
import os

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--lr', type = float, default = 1e-3, help='Please choose the learning rate')
    parser.add_argument('--batch_size', type = int, default = 1, help='Please choose the batch size')
    parser.add_argument('--epochs', type = int, default = 150, help='Please choose the number of epochs')
    parser.add_argument('--device', type = str, default = 2, help='Please choose the device')
    parser.add_argument('--dataset', type = str, default = 'mimic_500', help='Please choose the dataset')
    parser.add_argument('--model', type = str, default = None, help='Please choose the model')
    parser.add_argument('--beta1', type = float, default = 0.9, help='Please choose beta 1 for optimizer')
    parser.add_argument('--beta2', type = float, default = 0.99, help='Please choose beta 2 for optimizer')
    parser.add_argument('--eps', type = float, default = 1e-8, help='Please choose epsilon for optimizer')
    parser.add_argument('--warmup', type = int, default = 500, help = 'Please choose the number of warmup steps for the optimizer' )
    parser.add_argument('--weight_decay', type = float, default = 1e-2, help = 'Please choose the weight decay')
    parser.add_argument('--decay', type = float, default = 0.99, help='Please choose the decay') # 0.99 a bit smoother loss and perplexity (actually does not differ much) but reconstructions are near identical
    parser.add_argument('--seed', type = int, default = 0, help='Please choose the seed')
    parser.add_argument('--patience', type = int, default = 5, help='Please choose the patience')
    parser.add_argument('--dev', action = 'store_true', help = 'Please choose whether to use development mode or not')
    parser.add_argument('--inference', action = 'store_false', help = 'Please choose whether to inference or not') # Default is store true.
    parser.add_argument('--checkpoint', type = str, help = 'Please specify the checkpoint ')
    parser.add_argument('--log', action = 'store_true', help = 'Please choose whether to log or not')
    parser.add_argument('--dis', action = 'store_true', help = 'Please choose whether to distributed training or not')
    parser.add_argument('--tokenizer_check', type = str, help = 'Please specify the tokenizer')
    parser.add_argument('--num_merges', type = int, default = 1000, help = 'Please specify the vocab size') 
    parser.add_argument('--pad_to_max', type = int, default = 1000, help = 'Please specify the pad to max size') 
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU ids to use (e.g., "0,1,2")')
    parser.add_argument('--ports', type=str, default='12355', help='Comma-separated list of ports to use (e.g., "12355,12356,12357")')
    parser.add_argument('--toy', action = 'store_true', help = 'Please choose whether to use toy dataset or not')
    parser.add_argument('--peft', action = 'store_true', default = None, help = 'Please choose whether to use PEFT or not')
    parser.add_argument('--percentiles', type = str, default = None, help = 'Please choose the percentiles computed during preprocessing')
    parser.add_argument('--interpret', action = 'store_true', help = 'Please choose whether to interpret or not')
    return parser.parse_args()

def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.ports
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    
    args = get_args()
    device = torch.device(args.device)

    if args.dev:
        args.epochs=2
        
    # print('Loading API key')
    # with open('/mnt/newhome/atharva/projects/ECG-Byte/.huggingface/api_keys.txt', 'r') as file:
    #     file_contents = file.readlines()
    # api_key = file_contents[0].strip()

    # login(token = api_key)
    
    print('Collecting Garbage')
    gc.collect()
    print('Emptying CUDA Cache')
    torch.cuda.empty_cache()
    print(f'Setting Seed to {args.seed}')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    
    # if args.model == 'openai-community/gpt2-xl':
    #     target_modules = None # This automatically selects default modules
    # else:
    #     target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    
    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=target_modules,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type = TaskType.CAUSAL_LM,
    #     )
    


    # from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig
    # import torch

    # tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    # config = AutoConfig.from_pretrained("FacebookAI/roberta-base")
    # config.is_decoder = True
    # model = RobertaForCausalLM.from_pretrained("FacebookAI/roberta-base", config=config)

    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # outputs = model(**inputs)

    # prediction_logits = outputs.logits

    print('Initializing Model')
    # tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir = './../.huggingface')
    # llm = AutoModelForCausalLM.from_pretrained(args.model, cache_dir = './../.huggingface', torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    llm = AutoModelForCausalLM.from_pretrained("FacebookAI/roberta-base")
    
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
    # llm = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

    # if args.peft:
    #     llm = get_peft_model(llm, lora_config)
    #     llm.print_trainable_parameters()
        
    model = LLM(llm, args)
    model = model.to(device)
    model_hidden_size = model.llm.config.hidden_size
    find_unused_parameters = False
    
    # print(f'Total number of parameters: {count_parameters(model)}')
    
    # if args.dis:
    #     model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)
    
    print(f'Loading {args.dataset}')
    args.inference = False

    if args.inference:
        json_path = "/mnt/newhome/atharva/projects/DemosResearch/data/pretrain_mimic.json"

        test_data = ECGChatDataset(json_path,args,test_mode=True)
        test_loader = DataLoader(test_data,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True)
        
        print(f'Inferencing {args.checkpoint}')
        seeds = [0, 42, 123, 456, 789]
        all_seed_results = []
        for seed in seeds:
            print(f'Setting Seed to {seed}')
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            checkpoint = torch.load(f'./runs/{args.seed}/{args.checkpoint}/best_model.pth', map_location=args.device)
            
            model.load_state_dict(checkpoint['model'])
            print('Model Loaded')
            seed_results = tester(model, test_loader, tokenizer, args)
            print("Working")
            break
        
        print('Inference Complete')

    else:
        # json_path = "/mnt/newhome/atharva/projects/DemosResearch/data/pretrain_mimic.json"
        json_path = "./data/ecg_instruct_45k.json"

        training_data = ECGChatDataset(json_path,args,test_mode=False)
        validation_data = ECGChatDataset(json_path,args,test_mode=True)
        
        training_loader = DataLoader(training_data, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    pin_memory=True,
                                    # sampler = train_sampler
                                    )
        
        validation_loader = DataLoader(validation_data, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    pin_memory=True, 
                                    )

            
        # optimizer = ScheduledOptim(
        # Adam(filter(lambda x: x.requires_grad, model.parameters()),
        #     betas=(args.beta1, args.beta2), eps=args.eps, lr = args.lr, weight_decay=args.weight_decay), model_hidden_size, args.warmup)
        optimizer = AdamW(
            model.parameters(),
            lr = args.lr,
        )


        train_loss = []
        val_loss = []

        all_epochs = []

        directory_path = "./dump"
        directory_path = os.path.join(directory_path,"testchat")
        os.makedirs(directory_path,exist_ok=True)

        # try:

        for epoch in range(args.epochs):
            # all_epochs.append(epoch)

            train_dic = trainer(model, tokenizer, training_loader, optimizer, args, epoch, directory_path)
            train_loss.append(train_dic['average_loss'])
            print(f"Training - Epoch: {epoch+1}\nTrain Loss: {train_dic['average_loss']}")
            
            val_dic = validater(model, validation_loader, args, epoch)
            val_loss.append(val_dic['average_loss'])
            print(f"Validating - Epoch: {epoch+1}\nVal Loss: {val_dic['average_loss']}")

            # early_stop = early_stopping(val_loss, patience=args.patience, delta=0.01)
            # if early_stop:
            #     print('Validation loss has stopped decreasing. Early stopping...')
            #     break

            model_state_dict = model.state_dict()

            checkpoint = {
                'model': model_state_dict,
                'epoch': epoch
            }

            # Save the best model based on validation loss
            if val_dic['average_loss'] <= min(val_loss):
                torch.save(checkpoint, f'{directory_path}/best_model.pth')
                print(f"Best model saved at epoch: {epoch+1}")

            print('-----------------------------------------------------------')

        # except Exception as e:
        #     try:
        #         print(f"An error occurred: {e}")
        #         # Save the latest model checkpoint in case of a crash
        #         torch.save(checkpoint, f'{directory_path}/crash_model.pth')
        #         print("Model checkpoint saved due to crash.")
        #     except:
        #         print("Skipping except!")
        # finally:
        #     try:
        #         torch.save(checkpoint, f'{directory_path}/crash_model.pth')
        #         print("Final attempt to save checkpoint due to crash or exit.")
        #     except:
        #         print("Skipping finally!")
        #     print('Training Finished')

if __name__ == '__main__':
    args = get_args()
    rank = 0
    world_size = 1
    main(rank,world_size)