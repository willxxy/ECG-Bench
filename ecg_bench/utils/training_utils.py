import random
import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel, \
                            ViTForMaskedImageModeling, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from transformers import logging
logging.set_verbosity_error()
import nltk
nltk.download('wordnet', quiet = True)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from evaluate import load
import numpy as np
from scipy import stats
from torch.optim import Adam, AdamW
import random, re, pathlib
from typing import List, Dict, Tuple, Callable, Optional
from collections import defaultdict

class TrainingUtils:
    def __init__(self, args, fm, viz, device, ecg_tokenizer_utils=None):
        self.args, self.fm, self.viz, self.device = args, fm, viz, device
        self.ecg_tokenizer_utils = ecg_tokenizer_utils
        self.cache_dir = "../.huggingface"

    def get_lora_configs(self):
        if 'gpt2' in self.args.model:
            target_modules = None # This automatically selects default modules
        else:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        lora_config = LoraConfig(
            r = self.args.lora_rank,
            lora_alpha = self.args.lora_alpha,
            target_modules = target_modules,
            task_type = TaskType.CAUSAL_LM,
            lora_dropout = self.args.lora_dropout,
            bias = "none")
        return lora_config
    
    def get_optimizer_class(self, optimizer_name):
        """Get optimizer class by name"""
        optimizers = {
            'adam': Adam,
            'adamw': AdamW,
        }
        if optimizer_name.lower() not in optimizers:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported: {list(optimizers.keys())}")
        return optimizers[optimizer_name.lower()]
    
    def create_model(self):
        if self.args.train == 'end2end' or self.args.inference == 'end2end':
            return self.get_llm()
        elif self.args.train == 'first' and self.args.inference == None: # since we only train, no inference
            return self.get_encoder()
        elif self.args.train == 'second' or self.args.inference == 'second':
            return self.get_llm_encoder()
    
    def get_llm_encoder(self):
        encoder_params = self.get_encoder()
        llm_params = self.get_llm()
        
        if any(key in self.args.model for key in ('vit', 'siglip', 'dinov2')):
            projection_dim = 768
        elif any(key in self.args.model for key in ('stmem', 'mtae', 'mlae')):
            projection_dim = 256
        elif 'clip' in self.args.model:
            projection_dim = 512
        elif 'merl' in self.args.model:
            projection_dim = 2048
        elif 'encoderfree' in self.args.model:
            projection_dim = self.args.seg_len * 12 # num leads
        if 'encoderfree' in self.args.model:
            from ecg_bench.models.encoder_llm.encoder_free_style import EncoderFree
            llava = EncoderFree(llm_params['llm'], projection_dim, llm_params['llm_tokenizer']).to(self.device)
        else:
            from ecg_bench.models.encoder_llm.llava_style import LLaVA
            llava = LLaVA(llm_params['llm'], encoder_params['encoder'], 
                        projection_dim, llm_params['llm_tokenizer']).to(self.device)
        return_dict = {**encoder_params, **llm_params}
        return_dict['llava'] = llava
        return_dict['find_unused_parameters'] = False
        return return_dict
    
    def get_model_hidden_size(self, model):
        """
        Helper function to get hidden size from different model configurations.
        Handles various model architectures including Gemma3.
        """
        config = model.llm.config
        
        # For Gemma3 models, check text_config first
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            return config.text_config.hidden_size
        # For regular models, check hidden_size directly
        elif hasattr(config, 'hidden_size'):
            return config.hidden_size
        # For some models, it might be in different attributes
        elif hasattr(config, 'd_model'):
            return config.d_model
        elif hasattr(config, 'n_embd'):
            return config.n_embd
        else:
            # Fallback: try to infer from model structure
            if hasattr(model.llm, 'embed_tokens') and hasattr(model.llm.embed_tokens, 'embedding_dim'):
                return model.llm.embed_tokens.embedding_dim
            else:
                raise AttributeError(f"Could not determine hidden size for model type: {type(config)}")
    
    def get_llm(self):
        # Model configuration mapping
        model_configs = {
            'llama': {
                'import_path': 'ecg_bench.models.llm.llama',
                'class_name': 'llama',
                'hf_path': 'meta-llama'
            },
            'opt': {
                'import_path': 'ecg_bench.models.llm.opt',
                'class_name': 'opt',
                'hf_path': 'facebook'
            },
            'gpt2': {
                'import_path': 'ecg_bench.models.llm.gpt2',
                'class_name': 'gpt2',
                'hf_path': 'openai-community'
            },
            'gemma': {
                'import_path': 'ecg_bench.models.llm.gemma',
                'class_name': 'gemma',
                'hf_path': 'google'
            },
            'qwen': {
                'import_path': 'ecg_bench.models.llm.qwen',
                'class_name': 'qwen',
                'hf_path': 'qwen'
            }
        }

        model_key = next((key for key in model_configs.keys() if key.lower() in self.args.model.lower()), None)
        if not model_key:
            raise ValueError(f"Unsupported model: {self.args.model}")
        
        config = model_configs[model_key]
        
        module = __import__(config['import_path'], fromlist=[config['class_name']])
        model_class = getattr(module, config['class_name'])
        
        if self.args.train == 'second' or self.args.inference == 'second':
            llm_model_name = self.args.model.split('_')[1]
        else:
            llm_model_name = self.args.model
        
        hf_llm = AutoModelForCausalLM.from_pretrained(
            f"{config['hf_path']}/{llm_model_name}",
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.args.attn_implementation,
        ).to(self.device)
        
        llm_tokenizer = AutoTokenizer.from_pretrained(
            f"{config['hf_path']}/{llm_model_name}",
            cache_dir=self.cache_dir
        )
    
        if self.args.train == 'end2end' or self.args.inference == 'end2end':
            vocab_keys = list(self.ecg_tokenizer_utils.vocab.keys())
        elif self.args.train == 'second' or self.args.inference == 'second':
            vocab_keys = None
        
        llm, llm_tokenizer = self.modify_llm_tokenizer(
                hf_llm, 
                llm_tokenizer, 
                vocab_keys
            )
        
        if self.args.peft:
            llm = get_peft_model(llm, self.get_lora_configs())
            llm.print_trainable_parameters()
        
        llm = model_class(llm, self.args).to(self.device)
        
        return {
            'llm': llm,
            'llm_tokenizer': llm_tokenizer,
            'find_unused_parameters': False,
            'model_hidden_size': self.get_model_hidden_size(llm),
            'strict': True
        }
        
    def get_encoder(self):
        if 'clip' in self.args.model:
            from ecg_bench.models.encoder.clip import CLIP
            hf_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = self.cache_dir).to(self.device)
            encoder = CLIP(hf_encoder).to(self.device)
            encoder_tokenizer = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir = self.cache_dir)
            find_unused_parameters = False
            model_hidden_size = encoder.clip.config.projection_dim
            strict = True
        elif 'siglip' in self.args.model:
            from ecg_bench.models.encoder.siglip import SIGLIP
            hf_encoder = AutoModel.from_pretrained("google/siglip-base-patch16-224", cache_dir = self.cache_dir).to(self.device)
            encoder = SIGLIP(hf_encoder).to(self.device)
            encoder_tokenizer = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", cache_dir = self.cache_dir, use_fast = True)
            find_unused_parameters = False
            model_hidden_size = encoder.siglip.config.text_config.hidden_size
            strict = True
        elif 'vit' in self.args.model:
            from ecg_bench.models.encoder.vit import ViT   
            hf_encoder = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = self.cache_dir).to(self.device) 
            encoder = ViT(hf_encoder).to(self.device)
            encoder_tokenizer = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = self.cache_dir, use_fast = True)
            find_unused_parameters = False
            model_hidden_size = encoder.vit.config.hidden_size
            self.args.num_patches = (encoder.vit.config.image_size // encoder.vit.config.patch_size) ** 2
            strict = True
        elif 'dinov2' in self.args.model:
            from ecg_bench.models.encoder.dinov2 import DINOv2
            hf_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base", cache_dir = self.cache_dir).to(self.device)
            encoder = DINOv2(hf_encoder).to(self.device)
            encoder_tokenizer = AutoImageProcessor.from_pretrained("facebook/dinov2-base", cache_dir = self.cache_dir)
            find_unused_parameters = False
            model_hidden_size = encoder.dinov2.config.hidden_size
        elif 'merl' in self.args.model:
            from ecg_bench.models.encoder.merl import MERL, MERLPretrain
            lm, encoder_tokenizer = self.get_lm('ncbi/MedCPT-Query-Encoder')
            encoder = MERLPretrain('resnet101', lm, self.args, self.device).to(self.device)
            encoder = MERL(encoder).to(self.device)
            find_unused_parameters = True
            model_hidden_size = 256
            strict = False
        elif 'stmem' in self.args.model:
            from ecg_bench.models.encoder.st_mem import st_mem_vit_base_dec256d4b, ST_MEM_Ours
            encoder = st_mem_vit_base_dec256d4b(device=self.device, num_leads=12, seq_len=self.args.seg_len, patch_size=self.calculate_patch_size(self.args.seg_len)).to(self.device)
            encoder = ST_MEM_Ours(encoder).to(self.device)
            find_unused_parameters = False # first
            model_hidden_size = 768
            strict = False
            encoder_tokenizer = None
        elif 'mtae' in self.args.model:
            from ecg_bench.models.encoder.mtae import mtae_vit_base_dec256d4b, MTAE_Ours
            encoder = mtae_vit_base_dec256d4b(device=self.device, num_leads=12, seq_len=self.args.seg_len, patch_size=self.calculate_patch_size(self.args.seg_len)).to(self.device)
            encoder = MTAE_Ours(encoder).to(self.device)
            find_unused_parameters = False # first
            model_hidden_size = 768
            strict = False
            encoder_tokenizer = None
        elif 'mlae' in self.args.model:
            from ecg_bench.models.encoder.mlae import mlae_vit_base_dec256d4b, MLAE_Ours
            # THIS DOES PATCHES BY LEADS
            encoder = mlae_vit_base_dec256d4b(device=self.device, num_leads=12, seq_len=self.args.seg_len, patch_size=1).to(self.device)
            encoder = MLAE_Ours(encoder).to(self.device)
            find_unused_parameters = False # first
            model_hidden_size = 768
            strict = False
            encoder_tokenizer = None
        elif 'encoderfree' in self.args.model:
            encoder = None
            find_unused_parameters = False # first
            model_hidden_size = None
            strict = False
            encoder_tokenizer = None
        if self.args.encoder_checkpoint != None:
            encoder_checkpoint = torch.load(f'{self.args.encoder_checkpoint}/best_model.pth', map_location = self.device)
            encoder.load_state_dict(encoder_checkpoint['model'], strict = strict)
            encoder = encoder.to(device = self.device, dtype = torch.bfloat16) ### dtype of llm
        
        return {
            'encoder': encoder,
            'encoder_tokenizer': encoder_tokenizer,
            'find_unused_parameters': find_unused_parameters,
            'model_hidden_size': model_hidden_size,
            'strict': strict
        }
    def calculate_patch_size(self, seq_len):
        min_patches = 16
        max_patches = 64        
        factors = [i for i in range(1, seq_len + 1) if seq_len % i == 0]
        patch_candidates = []
        for patch_size in factors:
            num_patches = seq_len // patch_size
            if min_patches <= num_patches <= max_patches:
                patch_candidates.append(patch_size)
        if not patch_candidates:
            target = int(np.sqrt(seq_len/32))
            patch_size = min(factors, key=lambda x: abs(x - target))
        else:
            patch_size = min(patch_candidates, 
                           key=lambda x: abs(seq_len//x - 32))
        return patch_size
        
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
    
    def modify_llm_tokenizer(self, llm, llm_tokenizer, new_ids = None):
        # Regular tokens for signal IDs
        if new_ids is not None:
            new_ids = [f'signal_{str(ids)}' for ids in new_ids]
            llm_tokenizer.add_tokens(new_ids)
        
        # Special tokens added in more idiomatic way
        special_tokens = {
            'additional_special_tokens': [],
            'pad_token': '<pad>' # IS PAD TOKEN DIFFERENT FOR EACH LLM?
        }
        if self.args.train == 'second' or self.args.inference == 'second':
            special_tokens['additional_special_tokens'].append('<signal>')
        
        ### THIS IS FOR LLAMA
        if 'llama' in self.args.model:
            special_tokens['additional_special_tokens'].append('<|start_header_id|>')
            special_tokens['additional_special_tokens'].append('<|end_header_id|>')
            special_tokens['additional_special_tokens'].append('<|eot_id|>')
        llm_tokenizer.add_special_tokens(special_tokens)
        llm.config.pad_token_id = llm_tokenizer.pad_token_id
        llm.resize_token_embeddings(len(llm_tokenizer))
        return llm, llm_tokenizer
    
    def calculate_acc(self, references, hypotheses):
        return np.mean([ref == hyp for ref, hyp in zip(references, hypotheses)])
    
    def calculate_bleu(self, references, hypotheses):
        smoother = SmoothingFunction()
        # Convert to list of lists for corpus_bleu
        references = [[r.split()] for r in references]
        hypotheses = [h.split() for h in hypotheses]
        return corpus_bleu(references, hypotheses, smoothing_function=smoother.method1)

    def calculate_meteor(self, references, hypotheses):
        # Calculate METEOR for the entire corpus at once
        scores = [meteor_score([ref.split()], hyp.split()) for ref, hyp in zip(references, hypotheses)]
        return np.mean(scores)

    def calculate_rouge(self, references, hypotheses):
        rouge = Rouge()
        # Calculate ROUGE for all pairs at once
        scores = rouge.get_scores(hypotheses, references, avg=True)
        return {
            'rouge-1': scores['rouge-1']['f'],
            'rouge-2': scores['rouge-2']['f'],
            'rouge-l': scores['rouge-l']['f']
        }

    def calculate_bertscore(self, references, hypotheses, device):
        bertscore = load('bertscore')
        # Calculate BERTScore for all pairs at once
        results = bertscore.compute(
            predictions=hypotheses,
            references=references,
            lang='en',
            device=device
        )
        return {
            'hf-prec': results['precision'],
            'hf-rec': results['recall'],
            'hf-f1': results['f1']
        }

    def evaluate_strings(self, references, hypotheses, device):
        if len(references) != len(hypotheses):
            raise ValueError("The number of references and hypotheses must be the same.")
        
        # Filter out empty strings to avoid evaluation errors
        valid_pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses) if ref and hyp]
        if not valid_pairs:
            return {
                'BLEU': 0,
                'METEOR': 0.0,
                'ROUGE': {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0},
                'BERTSCORE': {'hf-prec': [0.0], 'hf-rec': [0.0], 'hf-f1': [0.0]},
                'ACC': 0.0
            }
            
        valid_refs, valid_hyps = zip(*valid_pairs)
        
        return {
            'BLEU': self.calculate_bleu(valid_refs, valid_hyps),
            'METEOR': self.calculate_meteor(valid_refs, valid_hyps),
            'ROUGE': self.calculate_rouge(valid_refs, valid_hyps),
            'BERTSCORE': self.calculate_bertscore(valid_refs, valid_hyps, device),
            'ACC': self.calculate_acc(valid_refs, valid_hyps)
        }
        

    def run_statistical_analysis(self, all_seeds_results):
        metrics = list(all_seeds_results[0]['metrics'].keys())
        statistical_results = {}
        
        for metric in metrics:
            # Get the metric values from all seeds
            metric_values = [result['metrics'][metric] for result in all_seeds_results]
            
            # Handle dictionary metrics (ROUGE and BERTSCORE)
            if isinstance(metric_values[0], dict):
                statistical_results[metric] = {}
                # Calculate statistics for each sub-metric
                for sub_metric in metric_values[0].keys():
                    if isinstance(metric_values[0][sub_metric], list):
                        # Handle BERTScore which returns lists
                        values = [np.mean(result['metrics'][metric][sub_metric]) * 100 for result in all_seeds_results]
                    else:
                        # Handle ROUGE scores which are single values
                        values = [result['metrics'][metric][sub_metric] * 100 for result in all_seeds_results]
                    
                    mean = np.mean(values)
                    std = np.std(values, ddof=1)
                    
                    confidence = 0.95
                    degrees_of_freedom = len(values) - 1
                    t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
                    margin_of_error = t_value * (std / np.sqrt(len(values)))
                    
                    conf_interval = (mean - margin_of_error, mean + margin_of_error)
                    
                    statistical_results[metric][sub_metric] = {
                        'mean': mean,
                        'std': std,
                        'conf_interval': conf_interval,
                        'raw_values': values
                    }
            else:
                # Handle single value metrics (BLEU and METEOR)
                values = [result['metrics'][metric] * 100 for result in all_seeds_results]
                
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                
                confidence = 0.95
                degrees_of_freedom = len(values) - 1
                t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
                margin_of_error = t_value * (std / np.sqrt(len(values)))
                
                conf_interval = (mean - margin_of_error, mean + margin_of_error)
                
                statistical_results[metric] = {
                    'mean': mean,
                    'std': std,
                    'conf_interval': conf_interval,
                    'raw_values': values
                }
        
        return statistical_results
    
    def get_lm(self, hf_model_name):
        ### lm is language model (different from llm) typically used for text encoder
        lm = AutoModel.from_pretrained(hf_model_name, cache_dir = self.cache_dir)
        lm_tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir = self.cache_dir)
        return lm, lm_tokenizer