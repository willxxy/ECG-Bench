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
        
        # Project ECG signal to LLM hidden dimension
        signal_embeddings = self.signal_projection(flattened_signal)  # [batch_size, hidden_size]
        
        # Get token embeddings from LLM
        token_embeddings = self.llm.get_input_embeddings()(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # Replace <signal> token embedding with projected ECG signal
        for batch_idx in range(batch_size):
            if isinstance(signal_id_index, list):
                sig_idx = signal_id_index[batch_idx]
            else:
                sig_idx = signal_id_index
            token_embeddings[batch_idx, sig_idx] = signal_embeddings[batch_idx]
        
        # Forward through LLM with modified embeddings
        outputs = self.llm(
            inputs_embeds=token_embeddings,
            attention_mask=attn_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, input_ids, attn_mask, encoder_out, signal_id_index, **generation_kwargs):
        """
        Generation method for inference.
        """
        batch_size = input_ids.size(0)
        
        # Get ECG signal embeddings
        ecg_signal = encoder_out['signal']
        flattened_signal = ecg_signal.view(batch_size, -1)
        signal_embeddings = self.signal_projection(flattened_signal)
        
        # Get token embeddings
        token_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        # Replace <signal> token embedding
        for batch_idx in range(batch_size):
            if isinstance(signal_id_index, list):
                sig_idx = signal_id_index[batch_idx]
            else:
                sig_idx = signal_id_index
            token_embeddings[batch_idx, sig_idx] = signal_embeddings[batch_idx]
        
        # Generate using the LLM
        with torch.no_grad():
            generated_ids = self.llm.generate(
                inputs_embeds=token_embeddings,
                attention_mask=attn_mask,
                **generation_kwargs
            )
        
        return generated_ids
    
    def signal_to_image_for_fuyu(self, ecg_signal):
        """
        Convert ECG signal to image format that Fuyu can process.
        This is mainly for compatibility with Fuyu's native image processing capabilities.
        """
        # Normalize signal for visualization
        normalized_signal = (ecg_signal - ecg_signal.min()) / (ecg_signal.max() - ecg_signal.min())
        
        # Create RGB image representation
        # Shape: [num_leads, seq_len] -> [height, width, 3]
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


# Integration with the existing training utilities
def create_fuyu8b_model(args, device, cache_dir='../.huggingface'):
    """
    Factory function to create Fuyu-8B model following the existing pattern.
    This should be integrated into the TrainingUtils.get_llm_encoder() method.
    """
    # Load Fuyu-8B model and processor
    model_id = "adept/fuyu-8b"
    hf_fuyu = FuyuForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).to(device)
    
    # Calculate projection dimension for encoder-free approach
    projection_dim = args.seg_len * 12  # seq_len * num_leads
    
    # Create Fuyu8B instance
    fuyu_model = Fuyu8B(
        llm=hf_fuyu,
        encoder=None,  # Encoder-free
        projection_dim=projection_dim,
        tokenizer=None  # Will be set by training utils
    )
    
    # Wrap for compatibility with training pipeline
    wrapped_model = FuyuECGWrapper(fuyu_model, args)
    
    return wrapped_model


# Example usage in training loop
def example_training_integration():
    """
    Example of how this would integrate with the existing training pipeline.
    """
    # This would go in your main training script
    
    # Assuming you have args, device, and other setup from existing code
    # model = create_fuyu8b_model(args, device)
    
    # The model can then be used with existing SecondStageECGChatDataset
    # dataset = SecondStageECGChatDataset(json_data_file, train_utils, encoder_tokenizer, llm_tokenizer)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Training loop (similar to existing pattern)
    # for epoch in range(args.epochs):
    #     for batch in dataloader:
    #         outputs = model(batch)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    
    pass