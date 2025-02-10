from ecg_bench.models.encoder.mtae import mtae_vit_base_dec256d4b
import torch

batch_size = 4
x = torch.randn(batch_size, 12, 500)

model = mtae_vit_base_dec256d4b(num_leads=12, seq_len=500, patch_size=20)

out = model(x)
print(out['loss'])