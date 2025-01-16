import random
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, \
                            ViTForMaskedImageModeling, AutoModel, AutoTokenizer
class TrainingUtils:
    def __init__(self, args, fm, viz):
        self.args = args
        self.fm = fm
        self.viz = viz
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
            encoder_tokenizer = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir = self.cache_dir)
            find_unused_parameters = False
            model_hidden_size = encoder.vit.config.hidden_size
            self.args.num_patches = (encoder.vit.config.image_size // encoder.vit.config.patch_size) ** 2
        
        return {
            'encoder': encoder,
            'encoder_tokenizer': encoder_tokenizer,
            'encoder2': encoder2,
            'encoder2_tokenizer': encoder_tokenizer2,
            'llm': llm,
            'llm_tokenizer': llm_tokenizer,
            'find_unused_parameters': find_unused_parameters,
            'model_hidden_size': model_hidden_size
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
        