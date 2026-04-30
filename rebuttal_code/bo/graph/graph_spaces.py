import numpy as np
import torch


class LowRankGraphSpace:
    def __init__(self, nodes, rank, tau, device=None):
        self.nodes = int(nodes)
        self.rank = int(rank)
        self.tau = float(tau)
        self.dim = 2 * self.nodes * self.rank

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _to_torch(self, z):
        if isinstance(z, torch.Tensor):
            z_t = z
        else:
            z_t = torch.as_tensor(z, dtype=torch.float32)
        return z_t.to(self.device)

    def vec2score(self, z):
        z_t = self._to_torch(z)
        single = (z_t.ndim == 1)
        if single:
            z_t = z_t[None, :]

        n = self.nodes * self.rank
        zA = z_t[..., :n]
        zB = z_t[..., n:2 * n]

        A_lr = zA.reshape(*zA.shape[:-1], self.nodes, self.rank)
        B_lr = zB.reshape(*zB.shape[:-1], self.nodes, self.rank)

        # S_ij = sum_k A_ik * B_jk
        S = torch.einsum("...ik,...jk->...ij", A_lr, B_lr)

        S_np = S.detach().to("cpu").numpy().astype(np.float32)
        return S_np[0] if single else S_np

    def vec2adj(self, z):
        S = self.vec2score(z)
        G = (S > self.tau).astype(np.int32)
        return G
