from torch import nn
import torch


class Signal2Vec(nn.Module):
    def __init__(self, builder, projection_dim, llm_id):
        super().__init__()
        self.llm = llm_id
        w = builder.load_embeddings(builder.args.signal2vec_embeddings)  # [V, D]
        V, D = w.size()
        assert D == projection_dim, f"Expected dim={projection_dim}, got {D}"
        self.embedding = nn.Embedding(V, D)
        with torch.no_grad():
            self.embedding.weight.copy_(w)
        self.embedding.weight.requires_grad_(False)

    @torch.no_grad()
    def get_encoder_embeddings(self, batch):
        # elm_input_ids = batch["elm_input_ids"]
        pass
