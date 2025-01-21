import torch
import torch.nn as nn

class LLaVA(nn.Module):
    def __init__(self, llm, encoder, projection_dim, llm_tokenizer):
        super(LLaVA, self).__init__()
        self.llm = llm
        self.encoder = encoder
        self.llm_tokenizer = llm_tokenizer
        for param in self.encoder.parameters():
            param.requires_grad = False ## do this in training_utils.py
        
        
        self.projection_dim = projection_dim
        self.encoder_projection = nn.Linear(self.projection_dim, self.llm.llm.config.hidden_size).to(dtype = self.llm.llm.dtype)
        
    def forward(self, batch):
        projected_embeds = self.get_projections(batch['encoder_out'])
        llm_embeddings = self.llm.get_llm_embeddings(batch['input_ids'])
        
        llm_embeddings = self.modify_llm_embeddings(
            projected_embeds, 
            llm_embeddings,
            batch['signal_id_index'],
            batch['input_ids']
        )
        
        outputs = self.llm(
            inputs_embeds=llm_embeddings,
            attention_mask=batch['attn_mask'],
            labels=batch['labels'],
            position_ids=batch['position_ids']
        )
        
        return outputs
    
    def get_projections(self, encoder_out):
        signal_embeds = self.encoder.get_embeddings(encoder_out)
        projected_embeds = self.encoder_projection(signal_embeds.to(dtype = self.llm.llm.dtype))
        return projected_embeds
    
    
    def modify_llm_embeddings(self, projected_embeds, llm_embeddings, signal_id_index, input_ids):
        # Replace the embedding at the signal token position in the current sequence
        llm_embeddings[:, signal_id_index, :] = projected_embeds
        
        # Update the embedding table for the <signal> token
        with torch.no_grad():
            signal_token_id = self.llm_tokenizer.convert_tokens_to_ids(['<signal>'])[0]  # Get the actual token ID for <signal>
            self.llm.llm.get_input_embeddings().weight[signal_token_id].copy_(projected_embeds.squeeze(1))
        
        return llm_embeddings