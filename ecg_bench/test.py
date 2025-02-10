from ecg_bench.models.encoder.st_mem import st_mem_vit_small, st_mem_vit_base, \
    st_mem_vit_small_dec256d4b, st_mem_vit_base_dec256d4b
import torch

batch_size = 4
x = torch.randn(batch_size, 12, 500)

model = st_mem_vit_small_dec256d4b(num_leads=12, seq_len=500, patch_size=20)

out = model(x)
print(out['x_latents'].shape)

model = st_mem_vit_base_dec256d4b(num_leads=12, seq_len=500, patch_size=20)

out = model(x)
print(out['x_latents'].shape)