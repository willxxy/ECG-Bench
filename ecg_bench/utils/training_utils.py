import random
import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, \
                            ViTForMaskedImageModeling, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

class TrainingUtils:
    def __init__(self, args, fm, viz, ecg_tokenizer_utils = None):
        self.args = args
        self.fm = fm
        self.viz = viz
        self.ecg_tokenizer_utils = ecg_tokenizer_utils
        self.cache_dir = '../.huggingface'
    
    def split_dataset(self, data, train_ratio=0.7):
        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        indices = list(range(n_samples))
        random.shuffle(indices)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        return train_data, test_data

    def get_lora_configs(self):
        if self.args.model == 'gpt2-xl':
            target_modules = None # This automatically selects default modules
        else:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        lora_config = LoraConfig(
            r = self.args.lora_rank,
            lora_alpha = self.args.lora_alpha,
            target_modules = target_modules,
            task_type = TaskType.CausalLM,
            lora_dropout = self.args.lora_dropout,
            bias = "none")
        return lora_config
    
    def create_model(self):
        encoder = None
        encoder_tokenizer = None
        encoder2 = None
        encoder_tokenizer2 = None
        llm = None
        llm_tokenizer = None
        
        if self.args.model == 'clip':
            from ecg_bench.models.ecg_encoder.clip import CLIP
            hf_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = self.cache_dir)
            encoder = CLIP(hf_encoder)
            encoder_tokenizer = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir = self.cache_dir)
            find_unused_parameters = False
            model_hidden_size = encoder.clip.config.projection_dim
            print("Encoder Configs", encoder.clip.config)
            print(f"Model Hidden Size: {model_hidden_size}")
            print(f"Find Unused Parameters: {find_unused_parameters}")
        
        elif self.args.model == 'vit':
            from ecg_bench.models.ecg_encoder.vit import ViT   
            hf_encoder = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = self.cache_dir) 
            encoder = ViT(hf_encoder)
            encoder_tokenizer = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = self.cache_dir, use_fast = True)
            find_unused_parameters = False
            model_hidden_size = encoder.vit.config.hidden_size
            self.args.num_patches = (encoder.vit.config.image_size // encoder.vit.config.patch_size) ** 2
        
        elif self.args.model == 'llama-3.2-1b':
            from ecg_bench.models.llm.llama import Llama
            hf_llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir = self.cache_dir, torch_dtype = torch.bfloat16)
            llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir = self.cache_dir)
            llm, llm_tokenizer = self.modify_llm_tokenizer(hf_llm, llm_tokenizer, list(self.ecg_tokenizer_utils.vocab.keys()))
            if self.args.peft:
                llm = get_peft_model(llm, lora_config = self.get_lora_configs())
                llm.print_trainable_parameters()
            llm = Llama(llm, self.args)
            model_hidden_size = llm.llm.config.hidden_size
            find_unused_parameters = False
        
        return {
            'encoder': encoder,
            'encoder_tokenizer': encoder_tokenizer,
            'encoder2': encoder2,
            'encoder2_tokenizer': encoder_tokenizer2,
            'llm': llm,
            'llm_tokenizer': llm_tokenizer,
            'find_unused_parameters': find_unused_parameters,
            'model_hidden_size': model_hidden_size,
        }
        
    def count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    def early_stopping(self, losses):
        if len(losses) < self.args.patience + 1:
            return False
        best_loss = min(losses[:-self.args.patience])
        current_loss = losses[-1]
        if current_loss > best_loss + self.args.delta:
            return True
        return False
    
    def modify_llm_tokenizer(self, llm, llm_tokenizer, new_ids):
        new_ids = [f'signal_{str(ids)}' for ids in new_ids]
        llm_tokenizer.add_tokens(new_ids)
        llm_tokenizer.add_tokens(['<sig_start>'], special_tokens=True)
        llm_tokenizer.add_tokens(['<sig_end>'], special_tokens=True)
        if self.args.train == 'second':
            llm_tokenizer.add_tokens(['<signal>'], special_tokens=True)
        llm_tokenizer.add_special_tokens({"pad_token":"<pad>"})
        llm.config.pad_token_id = llm_tokenizer.pad_token_id
        llm.resize_token_embeddings(len(llm_tokenizer))
        return llm, llm_tokenizer