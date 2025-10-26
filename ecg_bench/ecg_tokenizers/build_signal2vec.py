# ecg_bench/ecg_tokenizers/signal2vec.py

import argparse
import os
from collections import Counter
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

from ecg_bench.utils.file_manager import FileManager
from ecg_bench.ecg_tokenizers.build_ecg_tokenizers import BuildECGByte


class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, c: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        ce = self.center(c)
        pe = self.context(p)
        ne = self.context(n)

        pos = torch.sum(ce * pe, dim=1)
        neg = torch.sum(ce.unsqueeze(1) * ne, dim=2)

        loss = -F.logsigmoid(pos).mean() - F.logsigmoid(-neg).mean()
        return loss

    @torch.no_grad()
    def embeddings_center(self) -> torch.Tensor:
        return self.center.weight.detach().clone()

    @torch.no_grad()
    def embeddings_avg(self) -> torch.Tensor:
        return ((self.center.weight + self.context.weight) / 2).detach().clone()


class SkipGramDataset(Dataset):
    def __init__(self, sequences: List[List[int]], window_max: int, neg_k: int, noise_dist: np.ndarray):
        self.sequences = [s for s in sequences if len(s) > 1]
        self.window_max = window_max
        self.neg_k = neg_k
        self.alias_prob, self.alias_idx = self._alias_setup(noise_dist)

        estimated_total = sum(len(s) * window_max for s in self.sequences)
        self.max_samples = min(10_000_000, estimated_total)

        if hasattr(self, "dev_mode") and self.dev_mode:
            self.max_samples = min(100_000, self.max_samples)

        print(f"SkipGramDataset: {len(self.sequences)} sequences, ~{estimated_total:,} estimated pairs, using {self.max_samples:,} samples")

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx: int):
        # Randomly sample a sequence and generate a pair on-demand
        seq = random.choice(self.sequences)
        if len(seq) < 2:
            # Fallback for very short sequences
            c = p = seq[0] if seq else 0
        else:
            # Generate a random pair from this sequence
            i = np.random.randint(0, len(seq))
            c = seq[i]
            # Sample positive context within window
            w = np.random.randint(1, self.window_max + 1)
            left, right = max(0, i - w), min(len(seq), i + w + 1)
            context_indices = [j for j in range(left, right) if j != i]
            if context_indices:
                p = seq[np.random.choice(context_indices)]
            else:
                p = seq[np.random.randint(0, len(seq))]

        n = self._alias_draw(self.neg_k)
        mask = n == p
        if mask.any():
            n[mask] = self._alias_draw(mask.sum())
        return (
            torch.tensor(c, dtype=torch.long),
            torch.tensor(p, dtype=torch.long),
            torch.tensor(n, dtype=torch.long),
        )

    @staticmethod
    def _alias_setup(probs: np.ndarray):
        K = len(probs)
        q = np.asarray(probs, dtype=np.float64) * K
        J = np.zeros(K, dtype=np.int32)
        small, large = [], []
        for i, qi in enumerate(q):
            (small if qi < 1.0 else large).append(i)
        while small and large:
            s, l = small.pop(), large.pop()
            J[s] = l
            q[l] = (q[l] + q[s]) - 1.0
            (small if q[l] < 1.0 else large).append(l)
        for i in small:
            q[i] = 1.0
            J[i] = i
        for i in large:
            q[i] = 1.0
            J[i] = i
        return q, J

    def _alias_draw(self, n: int) -> np.ndarray:
        K = len(self.alias_prob)
        kk = np.random.randint(0, K, size=n)
        accept = np.random.rand(n) < self.alias_prob[kk]
        out = np.where(accept, kk, self.alias_idx[kk])
        return out.astype(np.int64)


class BuildSignal2Vec:
    def __init__(self, fm: FileManager, ecg_byte_builder: BuildECGByte, args: argparse.Namespace):
        self.args = args
        self.fm = fm
        self.ecg_byte_builder = ecg_byte_builder

        if hasattr(self.args, "sampled_file") and self.args.sampled_file:
            with open(self.args.sampled_file) as f:
                self.sampled_file_paths = [ln.strip() for ln in f if ln.strip()]

    def train(self):
        raw_sequences = self._load_token_sequences()

        vocab_size, freq = self._infer_vocab_and_freq(raw_sequences)

        min_count = int(getattr(self.args, "min_count", 0))
        if min_count > 0:
            keep = np.where(freq >= min_count)[0]
            keep_mask = np.zeros(vocab_size, dtype=bool)
            keep_mask[keep] = True
            sequences = [[tok for tok in seq if keep_mask[tok]] for seq in raw_sequences]
            freq = self._recount(sequences, vocab_size)
        else:
            sequences = raw_sequences

        subsample_t = float(getattr(self.args, "subsample_t", 0.0))
        if subsample_t and subsample_t > 0.0:
            total = int(freq.sum())
            rel = freq / max(1, total)
            denom = np.maximum(1e-12, rel)
            p_keep = np.minimum(1.0, np.sqrt(subsample_t / denom) + subsample_t / denom)
            sequences = [self._subsample_seq(seq, p_keep) for seq in sequences]
            freq = self._recount(sequences, vocab_size)

        sequences = [s for s in sequences if len(s) > 1]
        if not sequences:
            raise RuntimeError("All sequences became empty after filtering/subsampling. Relax thresholds.")

        neg_alpha = float(getattr(self.args, "neg_alpha", 0.75))
        noise_dist = self._build_noise_dist_from_freq(freq, neg_alpha)

        ds = SkipGramDataset(
            sequences=sequences,
            window_max=int(self.args.window_size),
            neg_k=int(self.args.neg_k),
            noise_dist=noise_dist,
        )
        # Add dev mode flag if available
        if hasattr(self.args, "dev") and self.args.dev:
            ds.dev_mode = True
        num_workers = int(getattr(self.args, "num_workers", 0))
        dl = DataLoader(
            ds,
            batch_size=int(self.args.batch_size),
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SkipGramNeg(vocab_size=vocab_size, dim=int(self.args.embedding_dim)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(self.args.lr), weight_decay=1e-5)
        steps_per_epoch = max(1, len(dl))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_per_epoch * int(self.args.epochs))

        grad_clip = float(getattr(self.args, "grad_clip", 0.0))
        renorm_every = int(getattr(self.args, "renorm_every", 0))

        model.train()
        step = 0
        for epoch in tqdm(range(1, int(self.args.epochs) + 1), desc="Training"):
            total = 0.0
            for c, p, n in dl:
                c, p, n = c.to(device, non_blocking=True), p.to(device, non_blocking=True), n.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = model(c, p, n)
                loss.backward()
                if grad_clip and grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                sched.step()
                total += loss.item() * c.size(0)
                step += 1

                if renorm_every and (step % renorm_every == 0):
                    with torch.no_grad():
                        model.center.weight.data = F.normalize(model.center.weight.data, p=2, dim=1)
                        model.context.weight.data = F.normalize(model.context.weight.data, p=2, dim=1)

            avg = total / max(1, len(ds))
            print(f"[epoch {epoch}/{self.args.epochs}] loss={avg:.6f}")

        self._save_embeddings(f"{self.args.save_path}/embeddings.pt", model, vocab_size, freq)
        print(f"Saved embeddings to: {self.args.save_path}/embeddings.pt")

        self.tsne_visualize(f"{self.args.save_path}/embeddings.pt", f"{self.args.save_path}/tsne.png", top_k=1000)
        print(f"Saved t-SNE plot to: {self.args.save_path}/tsne.png")

    @staticmethod
    def load_embeddings(path: str) -> torch.Tensor:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt["embeddings"]

    def tsne_visualize(self, emb_path: str, out_png: str, top_k: int = 1000, id_to_label: dict = None):
        ckpt = torch.load(emb_path, map_location="cpu", weights_only=False)
        emb = ckpt["embeddings"]
        freq = ckpt["freq"]
        V = emb.shape[0]
        top_k = int(min(top_k, V))

        order = np.argsort(-freq)[:top_k]
        X = F.normalize(emb[order], p=2, dim=1).cpu().numpy()

        print("Running t-SNE (perplexity≈min(30, top_k/4)) ...")
        perp = int(min(30, max(5, top_k // 4)))
        tsne = TSNE(n_components=2, perplexity=perp, init="random", learning_rate="auto")
        Y = tsne.fit_transform(X)

        labels = [str(int(i)) if id_to_label is None else id_to_label.get(int(i), str(int(i))) for i in order]
        plt.figure(figsize=(8, 6), dpi=160)
        plt.scatter(Y[:, 0], Y[:, 1], s=6)
        for x, y, lab in zip(Y[:, 0], Y[:, 1], labels):
            plt.annotate(lab, (x, y), fontsize=6, alpha=0.7)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

    def _load_token_sequences(self) -> List[List[int]]:
        seqs: List[List[int]] = []
        count = 0
        for fp in tqdm(self.sampled_file_paths, desc="Loading token sequences"):
            arr = self.fm.open_npy(fp)["ecg"]
            symbols, _ = self.ecg_byte_builder.ecg_to_symbol(arr)
            tokens: List[int] = self.ecg_byte_builder.encode(symbols)
            if tokens:
                seqs.append(tokens)
            if getattr(self.args, "dev", False):
                count += 1
                if count >= 20:
                    break
        if not seqs:
            raise RuntimeError("No token sequences were produced. Check your tokenizer inputs.")
        return seqs

    @staticmethod
    def _infer_vocab_and_freq(seqs: List[List[int]]) -> Tuple[int, np.ndarray]:
        vmax = 0
        cnt = Counter()
        for s in seqs:
            if s:
                vmax = max(vmax, max(s))
                cnt.update(s)
        vocab_size = vmax + 1
        freq = np.zeros(vocab_size, dtype=np.int64)
        for i, c in cnt.items():
            freq[i] = c
        return vocab_size, freq

    @staticmethod
    def _recount(seqs: List[List[int]], vocab_size: int) -> np.ndarray:
        cnt = Counter()
        for s in seqs:
            cnt.update(s)
        freq = np.zeros(vocab_size, dtype=np.int64)
        for i, c in cnt.items():
            if i < vocab_size:
                freq[i] = c
        return freq

    @staticmethod
    def _subsample_seq(tokens: List[int], p_keep: np.ndarray) -> List[int]:
        if not tokens:
            return tokens
        rnd = np.random.rand(len(tokens))
        return [tok for tok, r in zip(tokens, rnd) if r < p_keep[min(tok, len(p_keep) - 1)]]

    def _build_noise_dist_from_freq(self, freq: np.ndarray, alpha: float) -> np.ndarray:
        prob = np.power(freq.astype(np.float64), alpha)
        s = prob.sum()
        if s <= 0:
            prob[:] = 1.0 / len(prob)
        else:
            prob /= s
        self._last_freq = freq.astype(np.float64)
        return prob

    def _save_embeddings(self, path: str, model: SkipGramNeg, vocab_size: int, freq: np.ndarray):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        emb = model.embeddings_avg().cpu()
        torch.save(
            {"embeddings": emb, "vocab_size": vocab_size, "dim": int(emb.size(1)), "freq": freq.astype(np.float64)},
            path,
        )
