from torch import nn
import torch


class LLaVA(nn.Module):
    def __init__(self, llm: nn.Module, encoder: nn.Module, projection: nn.Module, update_encoder: bool = False):
        super(LLaVA, self).__init__()
        self.llm = llm
        self.encoder = encoder
        self.projection = projection
        self.update_encoder = update_encoder
        self._set_encoder_trainable(self.update_encoder)

    def _set_encoder_trainable(self, trainable: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, batch):
        projected_embeds = self.get_projections(batch)
        llm_embeddings = self.llm.get_llm_embeddings(batch["elm_input_ids"])
        llm_embeddings = self.inject_projected_embeds(llm_embeddings, projected_embeds, batch["signal_id_indices"])
        batch["elm_inputs_embeds"] = llm_embeddings
        out = self.llm(batch)
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        if self.update_encoder:
            self.encoder.train(mode)
        else:
            self.encoder.eval()
        return self

    def get_projections(self, batch):
        if self.update_encoder:
            signal_embeds = self.encoder.get_encoder_embeddings(batch)
        else:
            with torch.no_grad():
                signal_embeds = self.encoder.get_encoder_embeddings(batch)
        projected_embeds = self.projection.project(signal_embeds)
        return projected_embeds

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

    def generate(self, batch):
        projected_embeds = self.get_projections(batch)
        llm_embeddings = self.llm.get_llm_embeddings(batch["elm_input_ids"])
        llm_embeddings = self.inject_projected_embeds(llm_embeddings, projected_embeds, batch["signal_id_indices"])
        batch["elm_inputs_embeds"] = llm_embeddings
        out = self.llm.generate(batch)
        return out
