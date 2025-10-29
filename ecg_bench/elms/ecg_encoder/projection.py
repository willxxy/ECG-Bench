from torch import nn

from ecg_bench.configs.constants import HF_LLMS


class Projection(nn.Module):
    def __init__(self, projection_dim, llm_id):
        super(Projection, self).__init__()
        self.llm = llm_id
        self.projection = nn.Linear(projection_dim, HF_LLMS[self.llm]["model_hidden_size"]).to(dtype=HF_LLMS[self.llm]["native_dtype"])

    def forward(self, batch):
        projected_embeds = self.projection(batch["ecg_signal"].to(device=batch["ecg_signal"].device, dtype=HF_LLMS[self.llm]["native_dtype"]))
        return projected_embeds

    def project(self, signal_embeds):
        projected_embeds = self.projection(signal_embeds.to(dtype=HF_LLMS[self.llm]["native_dtype"]))
        return projected_embeds
