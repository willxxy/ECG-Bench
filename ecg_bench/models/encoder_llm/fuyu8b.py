import torch
import torch.nn as nn
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import numpy as np

class Fuyu8B(nn.Module):
    """
    Fuyu-8B implementation for ECG processing following encoder-free multimodal approach.
    This class integrates with the existing ECG benchmark architecture.
    """
    def __init__(self, llm: FuyuForCausalLM, encoder: None, projection_dim: int, tokenizer):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.encoder = encoder  # Should be None for encoder-free approach
        
        # Projection layer to map ECG signal dimensions to LLM hidden size
        # projection_dim should be seg_len * 12 (number of leads) for encoder-free
        self.signal_projection = nn.Linear(projection_dim, self.llm.config.hidden_size).to(dtype=self.llm.dtype)
        
        # Initialize Fuyu processor for handling multimodal inputs
        self.processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        
    def forward(self, input_ids, attn_mask, encoder_out, signal_id_index, labels=None, **kwargs):
        """
        Forward pass for training/inference.
        
        Args:
            input_ids: Tokenized input sequence
            attn_mask: Attention mask
            encoder_out: Dictionary containing ECG signal data
            signal_id_index: Index where <signal> token appears
            labels: Target labels for training
        """
        batch_size = input_ids.size(0)
        
        # Get ECG signal embeddings (encoder-free approach)
        ecg_signal = encoder_out['signal']  # Shape: [batch_size, seq_len, num_leads]
        
        # Flatten ECG signal for projection
        # From [batch_size, seq_len, num_leads] to [batch_size, seq_len * num_leads]
        flattened_signal = ecg_signal.view(batch_size, -1)
        
        signal_embeddings = self.signal_projection(flattened_signal)
        
        # Get token embeddings from LLM
        token_embeddings = self.llm.get_input_embeddings()(input_ids)  # [batch_size, seq_len, hidden_size]
        
        for batch_idx in range(batch_size):
            if isinstance(signal_id_index, list):
                sig_idx = signal_id_index[batch_idx]
            else:
                sig_idx = signal_id_index
            token_embeddings[batch_idx, sig_idx] = signal_embeddings[batch_idx]
        
        outputs = self.llm(
            inputs_embeds=token_embeddings,
            attention_mask=attn_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, input_ids, attn_mask, encoder_out, signal_id_index, **generation_kwargs):
        batch_size = input_ids.size(0)
        
        ecg_signal = encoder_out['signal']
        flattened_signal = ecg_signal.view(batch_size, -1)
        signal_embeddings = self.signal_projection(flattened_signal)
        
        token_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        # Replace <signal> token embedding
        for batch_idx in range(batch_size):
            if isinstance(signal_id_index, list):
                sig_idx = signal_id_index[batch_idx]
            else:
                sig_idx = signal_id_index
            token_embeddings[batch_idx, sig_idx] = signal_embeddings[batch_idx]
        
        with torch.no_grad():
            generated_ids = self.llm.generate(
                inputs_embeds=token_embeddings,
                attention_mask=attn_mask,
                **generation_kwargs
            )
        
        return generated_ids
    
    def signal_to_image_for_fuyu(self, ecg_signal):
        normalized_signal = (ecg_signal - ecg_signal.min()) / (ecg_signal.max() - ecg_signal.min())
        
        height, width = normalized_signal.shape
        rgb_signal = np.stack([normalized_signal * 255] * 3, axis=-1).astype(np.uint8)
        
        return Image.fromarray(rgb_signal)
    
    def prepare_multimodal_inputs(self, ecg_signals, text_prompts):
        """
        Prepare inputs in Fuyu's native multimodal format.
        This method can be used for alternative processing approaches.
        """
        images = []
        for signal in ecg_signals:
            image = self.signal_to_image_for_fuyu(signal)
            images.append(image)
        
        # Use Fuyu processor to handle multimodal inputs
        inputs = self.processor(
            images=images,
            text=text_prompts,
            return_tensors="pt",
            padding=True
        )
        
        return inputs


class FuyuECGWrapper(nn.Module):
    """
    Wrapper class that integrates Fuyu8B with the existing ECG benchmark training pipeline.
    This ensures compatibility with the existing training utilities and data loaders.
    """
    def __init__(self, fuyu_model, args):
        super().__init__()
        self.fuyu = fuyu_model
        self.args = args
        
    def forward(self, batch):
        """
        Forward pass that matches the expected interface from the training pipeline.
        """
        input_ids = batch.get('input_ids')
        attn_mask = batch.get('attn_mask') 
        encoder_out = batch.get('encoder_out')
        signal_id_index = batch.get('signal_id_index')
        labels = batch.get('labels')
        
        return self.fuyu.forward(
            input_ids=input_ids,
            attn_mask=attn_mask,
            encoder_out=encoder_out,
            signal_id_index=signal_id_index,
            labels=labels
        )
    
    def generate(self, batch, **generation_kwargs):
        """
        Generation method for inference.
        """
        input_ids = batch.get('input_ids')
        attn_mask = batch.get('attn_mask')
        encoder_out = batch.get('encoder_out')
        signal_id_index = batch.get('signal_id_index')
        
        return self.fuyu.generate(
            input_ids=input_ids,
            attn_mask=attn_mask,
            encoder_out=encoder_out,
            signal_id_index=signal_id_index,
            **generation_kwargs
        )