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
    optim_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='Optimizer type')
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
    optim_group.add_argument('--ref_global_bs', type=int, default=None)
    optim_group.add_argument('--grad_accum_steps', type=int, default=1)
    optim_group.add_argument('--scale_wd', type=str, default='none', choices=['none','inv_sqrt','inv_linear'])
    
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
    mode_group.add_argument('--train_encoder', action='store_true', default=None, help='Train encoder too')
    mode_group.add_argument('--interpret', action='store_true', default=None, help='Interpret mode')
    mode_group.add_argument('--rag', action='store_true', default=None, help='RAG mode')
    mode_group.add_argument('--rag_k', type=int, default=1, help='RAG k')
    mode_group.add_argument('--rag_prompt_mode', type=str, default='system_prompt', choices=['system_prompt', 'user_query'], help='How to integrate RAG results: in system prompt, user query')
    mode_group.add_argument('--retrieval_base', type=str, default='combined', choices=['signal', 'feature', 'combined'], help='Retrieval base for similarity calculation')
    mode_group.add_argument('--retrieved_information', type=str, default='combined', choices=['feature', 'report', 'combined'], help='Type of information to retrieve in output')
    mode_group.add_argument('--load_rag_db', type = str, default = None, help = 'Load a RAG database')
    mode_group.add_argument('--load_rag_db_idx', type = str, default = None, help = 'Load a RAG database index')
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

    
    ### Preprocessing
    preprocess_group = parser.add_argument_group('Preprocessing')
    preprocess_group.add_argument('--base_data', type = str, default = None, help = 'Base dataset to preprocess')
    preprocess_group.add_argument('--map_data', type = str, default = None, help = 'External dataset to map to base dataset')
    preprocess_group.add_argument('--num_cores', type = int, default = 12, help = 'Number of cores for parallel processing')
    preprocess_group.add_argument('--num_tok_samples', type = int, default = 300000, help = 'Number of samples for training the tokenizer')
    preprocess_group.add_argument('--random_sampling', action = 'store_true', default = False, help = 'Use random sampling')
    preprocess_group.add_argument('--sample_files', action = 'store_true', default = False, help = 'Sample files')
    preprocess_group.add_argument('--preprocess_files', action = 'store_true', default = False, help = 'Preprocess files')
    preprocess_group.add_argument('--toy', action = 'store_true', default = False, help = 'Create a toy dataset')
    preprocess_group.add_argument('--create_rag_db', action = 'store_true', default = None, help = 'Create a RAG database')
    preprocess_group.add_argument('--mix_data', type=str, default=None, help='Mix data: comma-separated list of JSON filenames')
    
    ### Token Distribution
    token_dist_group = parser.add_argument_group('Token Distribution')
    token_dist_group.add_argument('--list_of_paths', type=str, default=None, help='Path to the list of paths for token distribution analysis')
    
    ### Track Encoding
    track_enc_group = parser.add_argument_group('Track Encoding')
    track_enc_group.add_argument('--num_plots', type=int, default=2, help='Number of plots to generate for track encoding')
    
    ### Tokenizer
    tokenizer_group = parser.add_argument_group('Tokenizer')
    tokenizer_group.add_argument('--sampled_files', type=str, default=None, help='Path to the sampled ECG files for tokenizer training')
    tokenizer_group.add_argument('--train_tokenizer', action='store_true', default=None, help='Train a new tokenizer')
    
    
    return parser.parse_args() 