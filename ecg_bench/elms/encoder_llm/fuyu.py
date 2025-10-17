from torch import nn
import torch


class Fuyu(nn.Module):
    def __init__(self, llm: nn.Module, encoder: nn.Module):
        super(Fuyu, self).__init__()
        self.encoder = encoder
        self.llm = llm

    def forward(self, batch):
        encoder_out = self.encoder(batch)
        llm_embeddings = self.llm.get_llm_embeddings(batch["elm_input_ids"])
        batch_indices = torch.arange(llm_embeddings.shape[0], device=llm_embeddings.device)
        llm_embeddings[batch_indices, batch["signal_id_indices"]] = encoder_out
        batch["elm_inputs_embeds"] = llm_embeddings
        out = self.llm(batch)
        return out
