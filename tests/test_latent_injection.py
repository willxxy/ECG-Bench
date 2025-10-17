import torch
import random


def inject_projected_embeds(llm_embeddings, projected_embeds, signal_id_indices):
    """
    llm_embeddings:   (B, T, D)
    projected_embeds: (B, D)      or (B, N, D)
    signal_id_indices:(B,) long   or (B, N) long
    """
    assert llm_embeddings.ndim == 3
    B, T, D = llm_embeddings.shape

    if projected_embeds.ndim == 2:
        projected_embeds = projected_embeds.unsqueeze(1)
    if signal_id_indices.ndim == 1:
        signal_id_indices = signal_id_indices.unsqueeze(1)

    assert projected_embeds.shape[:2] == signal_id_indices.shape, (
        f"shapes must match on (B, N): proj {projected_embeds.shape}, idx {signal_id_indices.shape}"
    )
    assert projected_embeds.shape[0] == B and projected_embeds.shape[2] == D
    assert signal_id_indices.dtype in (torch.int64, torch.long)
    assert (signal_id_indices >= 0).all() and (signal_id_indices < T).all(), "indices out of range"

    N = signal_id_indices.shape[1]
    device = llm_embeddings.device

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    flat_b = batch_idx.reshape(-1)
    flat_t = signal_id_indices.reshape(-1)
    flat_proj = projected_embeds.reshape(B * N, D)

    out = llm_embeddings.clone()
    out[flat_b, flat_t, :] = flat_proj
    return out


def test_single(B=4, T=9, D=7, device="cpu"):
    torch.manual_seed(1337)
    random.seed(1337)

    llm_embeddings = torch.randn(B, T, D, device=device)
    before = llm_embeddings.clone()

    projected = torch.randn(B, D, device=device)
    idx = torch.tensor([random.randrange(T) for _ in range(B)], device=device, dtype=torch.long)

    out = inject_projected_embeds(llm_embeddings, projected, idx)
    b = torch.arange(B, device=device)
    assert torch.allclose(out[b, idx], projected), "Injected slots != projected"

    mask = torch.ones((B, T), dtype=torch.bool, device=device)
    mask[b, idx] = False
    assert torch.allclose(out[mask], before[mask]), "Non-target positions changed"

    print(f"[OK single] device={device} projected={projected.shape} llm={llm_embeddings.shape} idx={idx.tolist()}")


def test_multi_uniform(B=3, T=8, D=5, N=2, device="cpu"):
    torch.manual_seed(0)

    emb = torch.randn(B, T, D, device=device)
    before = emb.clone()

    proj = torch.randn(B, N, D, device=device)
    idx = torch.tensor([[0, 3], [1, 5], [2, 6]], device=device)

    out = inject_projected_embeds(emb, proj, idx)

    for b in range(B):
        assert torch.allclose(out[b, idx[b]], proj[b]), f"Batch {b} injected slots mismatch"

    untouched = [(0, 1), (1, 2), (2, 4)]
    for b, t in untouched:
        assert torch.allclose(out[b, t], before[b, t]), f"Non-target (b={b}, t={t}) changed"

    print(f"[OK multi] device={device} proj={proj.shape} llm={emb.shape} idx={idx.tolist()}")


def main():
    test_single(device="cpu")
    test_multi_uniform(device="cpu")
    if torch.cuda.is_available():
        test_single(device="cuda")
        test_multi_uniform(device="cuda")


if __name__ == "__main__":
    main()
