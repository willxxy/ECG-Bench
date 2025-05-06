import argparse

def get_args():
    parser = argparse.ArgumentParser(description = None)
    
    ### Data
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data', type=str, default=None, help='Dataset name')
    data_group.add_argument('--seg_len', type=int, default=None, help='Segment length')
    data_group.add_argument('--num_merges', type=int, default=3500, help='Vocab size')
    data_group.add_argument('--target_sf', type=int, default=250, help='Target sampling frequency')
    data_group.add_argument('--pad_to_max', type=int, default=1024, help='Pad to max size')
    data_group.add_argument('--ecg_tokenizer', type=str, help='Tokenizer specification')
    data_group.add_argument('--percentiles', type=str, default=None, help='Percentiles computed during preprocessing')
    data_group.add_argument('--system_prompt', type=str, default=None, help='System prompt')
    data_group.add_argument('--image', action = 'store_true', default=None, help='Turn Image Generation on')
    data_group.add_argument('--augment_image', action = 'store_true', default=None, help='Turn Image Augmentation on')
    data_group.add_argument('--instance_normalize', action = 'store_true', default=True, help='Turn Instance Normalization on')
    data_group.add_argument('--perturb', action = 'store_true', default=None, help='Turn ECG Perturbation on')
    
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
    mode_group.add_argument('--train', type=str, default=None, choices=['first', 'second', 'end2end'], help='Training mode')
    mode_group.add_argument('--inference', type=str, default=None, choices=['second', 'end2end'], help='Inference mode')
    mode_group.add_argument('--post_train', action='store_true', default=None, help='Post-training mode')
    model_group.add_argument('--train_encoder', action='store_true', default=None, help='Train encoder too')
    mode_group.add_argument('--interpret', action='store_true', default=None, help='Interpret mode')
    mode_group.add_argument('--rag', action='store_true', default=None, help='RAG mode')
    mode_group.add_argument('--rag_k', type=int, default=1, help='RAG k')
    parser.add_argument('--load_rag_db', type = str, default = None, help = 'Load a RAG database')
    parser.add_argument('--load_rag_db_idx', type = str, default = None, help = 'Load a RAG database index')
    mode_group.add_argument('--dev', action='store_true', default=None, help='Development mode')
    mode_group.add_argument('--log', action='store_true', default=None, help='Enable logging')
    
    ### Distributed Training
    dist_group = parser.add_argument_group('Distributed Training')
    dist_group.add_argument('--dis', action='store_true', default=None, help='Enable distributed training')
    dist_group.add_argument('--gpus', type=str, default='0', help='Comma-separated GPU ids')
    dist_group.add_argument('--ports', type=str, default='12356', help='Comma-separated ports')
    
    ### Checkpoints
    ckpt_group = parser.add_argument_group('Checkpoints')
    ckpt_group.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    ckpt_group.add_argument('--encoder_checkpoint', type=str, default=None, help='Encoder checkpoint path')

    return parser.parse_args() 