import torch
import torch.nn as nn

class LLaVA(nn.Module):
    def __init__(self, llm, encoder, projection_dim, llm_tokenizer):
        super(LLaVA, self).__init__()
        self.llm = llm
        self.encoder = encoder
        self.llm_tokenizer = llm_tokenizer
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.projection_dim = projection_dim
        self.encoder_projection = nn.Linear(self.projection_dim, self.llm.llm.config.hidden_size).to(dtype=self.llm.llm.dtype)
            
    def forward(self, batch):
        projected_embeds = self.get_projections(batch['encoder_out'])
        llm_embeddings = self.llm.get_llm_embeddings(batch['input_ids'])
        llm_embeddings[:, batch['signal_id_index'], :] = projected_embeds
        batch['inputs_embeds'] = llm_embeddings
        out = self.llm(batch)
        return out
    
    def get_projections(self, encoder_out):
        signal_embeds = self.encoder.get_embeddings(encoder_out)
        projected_embeds = self.encoder_projection(signal_embeds.to(dtype=self.llm.llm.dtype))
        return projected_embeds
    
    def generate(self, batch, tokenizer):
        projected_embeds = self.get_projections(batch['encoder_out'])
        llm_embeddings = self.llm.get_llm_embeddings(batch['input_ids'])
        llm_embeddings[:, batch['signal_id_index'], :] = projected_embeds
        batch['inputs_embeds'] = llm_embeddings
        decoded_text = self.llm.generate(batch, tokenizer)
        return decoded_text 