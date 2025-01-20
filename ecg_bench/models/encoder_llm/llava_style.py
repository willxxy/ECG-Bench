import torch
import torch.nn as nn

class LLaVA(nn.Module):
    def __init__(self, llm, encoder, projection_dim):
        super(LLaVA, self).__init__()
        self.llm = llm
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False ## do this in training_utils.py
        
        
        self.projection_dim = projection_dim
        self.encoder_projection = nn.Linear(self.projection_dim, self.llm.llm.config.hidden_size).to(dtype = self.llm.llm.dtype)
        
        
        
    def forward(self, batch):
        projected_embeds = self.get_projections(batch)
        llm_embeddings = self.llm.get_llm_embeddings(batch)
        return out
    
    def get_projections(self, batch):
        signal_embeds = self.encoder.get_embeddings(batch)
        projected_embeds = self.encoder_projection(signal_embeds.to(dtype = self.llm.llm.dtype))
        return projected_embeds
    
    
    def prepare_input(self, projected_embeds, llm_embeddings, 
                      input_ids, attention_mask):
        