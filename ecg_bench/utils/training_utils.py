import random
import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel, \
                            ViTForMaskedImageModeling, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from transformers import logging
logging.set_verbosity_error()
import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from evaluate import load
import numpy as np
from scipy import stats


class TrainingUtils:
    def __init__(self, args, fm, viz, device, ecg_tokenizer_utils = None):
        self.args = args
        self.fm = fm
        self.viz = viz
        self.device = device
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
            task_type = TaskType.CAUSAL_LM,
            lora_dropout = self.args.lora_dropout,
            bias = "none")
        return lora_config
    
    def create_model(self):
        if self.args.train == 'end2end' or self.args.inference == 'end2end':
            return self.get_llm()
        elif self.args.train == 'first': # since we only train, no inference
            return self.get_encoder()
        elif self.args.train == 'second' or self.args.inference == 'second':
            return self.get_llm_encoder()
    
    def get_llm_encoder(self):
        encoder_params = self.get_encoder()
        llm_params = self.get_llm()
        return {**encoder_params, **llm_params}
    
    def get_llm(self):
        if self.args.model == 'llama-3.2-1b':
            from ecg_bench.models.llm.llama import Llama
            hf_llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir = self.cache_dir, 
                                                          torch_dtype = torch.bfloat16).to(self.device)
            llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir = self.cache_dir)
            llm, llm_tokenizer = self.modify_llm_tokenizer(hf_llm, llm_tokenizer, list(self.ecg_tokenizer_utils.vocab.keys()))
            if self.args.peft:
                llm = get_peft_model(llm, self.get_lora_configs())
                llm.print_trainable_parameters()
            llm = Llama(llm, self.args).to(self.device)
            model_hidden_size = llm.llm.config.hidden_size
            find_unused_parameters = False
            strict = True
            
        return {
            'llm': llm,
            'llm_tokenizer': llm_tokenizer,
            'find_unused_parameters': find_unused_parameters,
            'model_hidden_size': model_hidden_size,
            'strict': strict
        }
        
    def get_encoder(self):
        if self.args.model == 'clip':
            from ecg_bench.models.ecg_encoder.clip import CLIP
            hf_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = self.cache_dir)
            encoder = CLIP(hf_encoder)
            encoder_tokenizer = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir = self.cache_dir)
            find_unused_parameters = False
            model_hidden_size = encoder.clip.config.projection_dim
            strict = True
        
        elif self.args.model == 'vit':
            from ecg_bench.models.ecg_encoder.vit import ViT   
            hf_encoder = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = self.cache_dir) 
            encoder = ViT(hf_encoder)
            encoder_tokenizer = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = self.cache_dir, use_fast = True)
            find_unused_parameters = False
            model_hidden_size = encoder.vit.config.hidden_size
            self.args.num_patches = (encoder.vit.config.image_size // encoder.vit.config.patch_size) ** 2
            strict = True
        
        elif self.args.model == 'merl':
            from ecg_bench.utils.model_utils import MERLPretrain
            from ecg_bench.models.ecg_encoder.merl import MERL
            lm, encoder_tokenizer = self.get_lm('ncbi/MedCPT-Query-Encoder')
            encoder = MERLPretrain('resnet101', lm.to(self.device)).to(self.device)
            encoder = MERL(encoder, self.args).to(self.device)
            find_unused_parameters = True
            model_hidden_size = 256
            strict = False
        
        return {
            'encoder': encoder,
            'encoder_tokenizer': encoder_tokenizer,
            'find_unused_parameters': find_unused_parameters,
            'model_hidden_size': model_hidden_size,
            'strict': strict
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
                'BERTSCORE': {'hf-prec': [0.0], 'hf-rec': [0.0], 'hf-f1': [0.0]}
            }
            
        valid_refs, valid_hyps = zip(*valid_pairs)
        
        return {
            'BLEU': self.calculate_bleu(valid_refs, valid_hyps),
            'METEOR': self.calculate_meteor(valid_refs, valid_hyps),
            'ROUGE': self.calculate_rouge(valid_refs, valid_hyps),
            'BERTSCORE': self.calculate_bertscore(valid_refs, valid_hyps, device)
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