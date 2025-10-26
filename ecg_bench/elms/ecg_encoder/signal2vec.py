from torch import nn
import torch

from ecg_bench.configs.constants import HF_LLMS


class Signal2Vec(nn.Module):
    def __init__(self, signal2vec_builder, projection_dim, llm_id):
        super(Signal2Vec, self).__init__()
        self.llm = llm_id
        self.signal2vec_builder = signal2vec_builder
        self.signal2vec_embeddings = signal2vec_builder.load_embeddings(self.signal2vec_builder.args.signal2vec_embeddings)
        vocab_size = int(self.signal2vec_embeddings.size(0))
        self.embedding = nn.Embedding(vocab_size, projection_dim)
        with torch.no_grad():
            if (
                self.signal2vec_embeddings is not None
                and self.signal2vec_embeddings.size(0) == vocab_size
                and self.signal2vec_embeddings.size(1) == projection_dim
            ):
                self.embedding.weight.copy_(self.signal2vec_embeddings)
        self.embedding.weight.requires_grad_(False)

        self.projection = nn.Linear(projection_dim, HF_LLMS[self.llm]["model_hidden_size"]).to(dtype=HF_LLMS[self.llm]["native_dtype"])

    def forward(self, batch):
        symbols, _ = self.signal2vec_builder.ecg_byte_builder.ecg_to_symbol(batch["ecg_signal"])
        tokens = self.signal2vec_builder.ecg_byte_builder.encode(symbols)
        token_embeddings = self.embedding(tokens)
        projected_embeds = self.projection(token_embeddings)
        return projected_embeds
