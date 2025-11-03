from torch import nn
import torch


class Fuyu(nn.Module):
    def __init__(self, llm: nn.Module, encoder: nn.Module):
        super(Fuyu, self).__init__()
        self.encoder = encoder
        self.llm = llm

    def forward(self, batch):
        projected_embeds = self.get_projections(batch)
        llm_embeddings = self.llm.get_llm_embeddings(batch["elm_input_ids"])
        llm_embeddings = self.inject_projected_embeds(llm_embeddings, projected_embeds, batch["signal_id_indices"])
        batch["elm_inputs_embeds"] = llm_embeddings
        out = self.llm(batch)
        return out

    def generate(self, batch):
        projected_embeds = self.get_projections(batch)
        llm_embeddings = self.llm.get_llm_embeddings(batch["elm_input_ids"])
        llm_embeddings = self.inject_projected_embeds(llm_embeddings, projected_embeds, batch["signal_id_indices"])
        batch["elm_inputs_embeds"] = llm_embeddings
        out = self.llm.generate(batch)
        return out

    def get_projections(self, batch):
        return self.encoder(batch)

    def inject_projected_embeds(self, llm_embeddings: torch.Tensor, projected_embeds: torch.Tensor, signal_id_indices: torch.Tensor) -> torch.Tensor:
        assert llm_embeddings.ndim == 3
        B, T, H = llm_embeddings.shape

        if projected_embeds.ndim == 2:
            projected_embeds = projected_embeds.unsqueeze(1)
        if signal_id_indices.ndim == 1:
            signal_id_indices = signal_id_indices.unsqueeze(1)

        assert projected_embeds.shape[:2] == signal_id_indices.shape
        assert projected_embeds.shape[0] == B and projected_embeds.shape[2] == H
        assert (signal_id_indices >= 0).all() and (signal_id_indices < T).all()

        N = signal_id_indices.shape[1]
        dev = llm_embeddings.device
        batch_idx = torch.arange(B, device=dev).unsqueeze(1).expand(B, N)

        out = llm_embeddings.clone()
        out[batch_idx.reshape(-1), signal_id_indices.reshape(-1)] = projected_embeds.reshape(B * N, H)

        injected = out[batch_idx, signal_id_indices]
        assert torch.allclose(injected, projected_embeds, atol=1e-6), "Injection failed: projected embeddings not correctly written."

        return out
