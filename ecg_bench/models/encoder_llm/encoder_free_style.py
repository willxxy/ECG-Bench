import torch.nn as nn
import torch

class EncoderFree(nn.Module):
    def __init__(self, llm, projection_dim, llm_tokenizer):
        super(EncoderFree, self).__init__()
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        
        self.projection_dim = projection_dim
        self.encoder_projection = nn.Linear(self.projection_dim, self.llm.llm.config.hidden_size).to(dtype=self.llm.llm.dtype)
    
    def forward(self, batch):
        projected_embeds = self.get_projections(batch['encoder_out'])
        llm_embeddings = self.llm.get_llm_embeddings(batch['input_ids'])
        batch_size = llm_embeddings.shape[0]
        batch_indices = torch.arange(batch_size, device=llm_embeddings.device)
        llm_embeddings[batch_indices, batch['signal_id_index'], :] = projected_embeds
        batch['inputs_embeds'] = llm_embeddings
        out = self.llm(batch)
        return out

    def get_projections(self, encoder_out):
        signal_flat = encoder_out['signal'].reshape(encoder_out['signal'].shape[0], -1)
        projected_embeds = self.encoder_projection(signal_flat.to(device = self.encoder_projection.weight.device, dtype=self.llm.llm.dtype))
        return projected_embeds 
    
    def generate_chat(self, input_ids, attention_mask, tokenizer, encoder_out=None, signal_id_index=None):
        projected_embeds = self.get_projections(encoder_out)
        llm_embeddings = self.llm.get_llm_embeddings(input_ids)
        batch_size = llm_embeddings.shape[0]
        batch_indices = torch.arange(batch_size, device=llm_embeddings.device)
        llm_embeddings[batch_indices, signal_id_index, :] = projected_embeds
        out = self.llm.generate_chat(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            inputs_embeds=llm_embeddings
        )
        return out
