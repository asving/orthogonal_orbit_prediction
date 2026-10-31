"""Task, model, least-squares reference and a patchable forward pass.
Task: x_1 = U v, x_{t+1} = U x_t, U Haar-random orthogonal (n=6), v uniform on the unit sphere, T=24 tokens.
The model reads x_1..x_t and predicts x_{t+1} (MSE loss)."""
import torch, torch.nn as nn, torch.nn.functional as F, math
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
n, T = 6, 24

def haar(b):
    Q, R = torch.linalg.qr(torch.randn(b, n, n, device=dev))
    return Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)

def make_batch(b):
    U = haar(b); x = F.normalize(torch.randn(b, n, device=dev), dim=-1).unsqueeze(-1); xs = []
    for _ in range(T + 1): x = U @ x; xs.append(x)
    X = torch.cat(xs, -1).transpose(1, 2)
    return X[:, :T].contiguous(), X[:, 1:].contiguous(), U

class Block(nn.Module):
    def __init__(s, d, H):
        super().__init__(); s.d, s.H, s.dh = d, H, d // H
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

class TF(nn.Module):
    """Pre-LN transformer, d=128, 4 heads. use_mlp=False gives the attention-only variant."""
    def __init__(s, L, use_mlp=True, d=128, H=4):
        super().__init__(); s.L, s.use_mlp = L, use_mlp
        s.inp, s.pos = nn.Linear(n, d), nn.Parameter(torch.randn(T, d) * 0.02)
        s.blocks = nn.ModuleList([Block(d, H) for _ in range(L)])
        s.lnf, s.out = nn.LayerNorm(d), nn.Linear(d, n)
        s.register_buffer('mask', torch.triu(torch.ones(T, T, dtype=torch.bool), 1))

    def forward(s, X, ov=None):
        """ov: interventions. ('vsrc', l): fn(resid)->resid used ONLY to compute layer-l values;
        ('z', l): fn(z)->z on per-head attention outputs (B,H,T,dh); ('att', l): fn(att, resid)->att.
        Returns prediction; s.cache holds residuals 'resid' (embed + after each layer), 'z', 'att'."""
        ov = ov or {}; x = s.inp(X) + s.pos; s.cache = {'resid': [x], 'z': [], 'att': []}
        for l, b in enumerate(s.blocks):
            B_, Ln, d = x.shape
            qkv = lambda src: [t.transpose(1, 2) for t in b.qkv(b.ln1(src)).view(B_, Ln, 3, b.H, b.dh).unbind(2)]
            q, k, v = qkv(x)
            if ('vsrc', l) in ov: v = qkv(ov[('vsrc', l)](x))[2]
            att = ((q @ k.transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(s.mask, float('-inf')).softmax(-1)
            if ('att', l) in ov: att = ov[('att', l)](att, x)
            z = att @ v
            if ('z', l) in ov: z = ov[('z', l)](z)
            x = x + b.o(z.transpose(1, 2).reshape(B_, Ln, d))
            if s.use_mlp: x = x + b.mlp(b.ln2(x))
            s.cache['resid'].append(x); s.cache['z'].append(z); s.cache['att'].append(att)
        return s.out(s.lnf(x))

def load(name):
    L, mlp = int(name[1]), not name.endswith('attnonly')
    m = TF(L, mlp).to(dev); m.load_state_dict(torch.load(f'models/model_{name}.pt', map_location=dev)); return m.eval()

@torch.no_grad()
def least_squares(X):
    """Bayes-optimal predictor (Haar prior): pred_t = sum_k c_{t,k} x_{k+1}, c_t = pinv([x_1..x_{t-1}]) x_t. Returns pred, c."""
    B = X.shape[0]; pred, c = torch.zeros_like(X), torch.zeros(B, T, T, device=dev)
    for t in range(1, T):
        c[:, t, :t] = (torch.linalg.pinv(X[:, :t].transpose(1, 2)) @ X[:, t, :, None]).squeeze(-1)
        pred[:, t] = torch.einsum('bk,bkn->bn', c[:, t, :t], X[:, 1:t + 1])
    return pred, c

def corr(a, b):
    a, b = a.flatten() - a.mean(), b.flatten() - b.mean(); return ((a * b).sum() / (a.norm() * b.norm() + 1e-9)).item()
