"""Causal tests of the mechanism.   python causal.py [model name, e.g. L2, L3, L2_attnonly, L4_attnonly]
Mechanism: prediction = sum_j w_j x_j over (key x_{j-1}, value x_j) pairs made by a previous-token head, where the
least-squares weights are c_j = <x_{j-1}, S^+ x_t>, S = context covariance. Layer-0 similarity heads supply S x,
layer-1 heads attend with S-preconditioned dot products (truncated Neumann series for S^+), and later layers add
gradient-descent-like corrections that move the weights toward c.
Test 1  value-path patching: perturb x_j only where it is read as a VALUE -> causal weight w_j (least squares: w_j = c_j).
Test 2  query patching: donor sequence with the same x_t but a different context (reflected trajectory); patch the
        non-previous-token layer-0 heads at the query. First-order preconditioning predicts a shift along -M (S'-S) x_t.
Test 3  replace layer-1 attention by idealised patterns: pure dot product vs S-preconditioned bilinear form."""
import sys, torch, math
from common import *
torch.manual_seed(17); name = sys.argv[1] if len(sys.argv) > 1 else 'L2'; m = load(name); B = 1024
X, Y, _ = make_batch(B)
with torch.no_grad(): P0 = m(X); R0 = m.cache['resid']
ls, c = least_squares(X); s = torch.einsum('btn,bkn->btk', X, X)
print(f'##### {name}: late MSE vs least squares {((P0 - ls) ** 2).sum(-1)[:, n + 1:].mean():.4f}')

print('\n--- Test 1: value-path patching (causal pair weight w vs LS coefficient c and raw dot product s) ---')
eps = 0.2
for tq in [4, 12]:
    res = {}
    groups = [('layer 1', [1])] + ([('layers >= 2', list(range(2, m.L)))] if m.L > 2 else []) + [('all layers >= 1', list(range(1, m.L)))]
    for gname, Ls in groups:
        W, Cc, Ss = [], [], []
        for j in range(1, tq + 1):
            dl = torch.randn(B, n, device=dev); dl = F.normalize(dl - (dl * X[:, j]).sum(-1, keepdim=True) * X[:, j], dim=-1)
            Xd = X.clone(); Xd[:, j] = F.normalize(X[:, j] + eps * dl, dim=-1)
            with torch.no_grad():
                m(Xd); Rd = m.cache['resid']
                def mk(l, j=j):
                    def f(x): x = x.clone(); x[:, j] = Rd[l][:, j]; return x
                    return f
                Pp = m(X, {('vsrc', l): mk(l) for l in Ls})
            dv = Xd[:, j] - X[:, j]; W.append(((Pp[:, tq] - P0[:, tq]) * dv).sum(-1) / (dv * dv).sum(-1))
            Cc.append(c[:, tq, j - 1]); Ss.append(s[:, tq, j - 1])
        W, Cc, Ss = torch.stack(W, 1), torch.stack(Cc, 1), torch.stack(Ss, 1); res[gname] = W
        Fm = torch.stack([Cc.flatten(), Ss.flatten(), torch.ones_like(Ss.flatten())], -1); a = torch.linalg.lstsq(Fm, W.flatten()[:, None]).solution
        print(f'  t={tq:2d} {gname:16s} corr(w,c) {corr(W, Cc):+.3f}  corr(w,s) {corr(W, Ss):+.3f}   w ~ {a[0,0]:.2f} c + {a[1,0]:.2f} s')
    if m.L > 2:
        print(f'  t={tq:2d} gradient-step test: corr(w from layers>=2, c - w from layer 1) {corr(res["layers >= 2"], Cc - res["layer 1"]):+.3f}')

print('\n--- Test 2: query patching with a donor that has the same x_t but a different context ---')
with torch.no_grad(): m(X); PREV = int(torch.stack([m.cache['att'][0][:, h].diagonal(-1, -2, -1).mean() for h in range(4)]).argmax())
Xo, _, _ = make_batch(B)
for tq in [6, 12, 20]:
    u = F.normalize(Xo[:, tq] - X[:, tq], dim=-1); Hh = torch.eye(n, device=dev) - 2 * torch.einsum('bi,bj->bij', u, u)
    Xd = torch.einsum('bij,btj->bti', Hh, Xo)                       # reflected donor trajectory, Xd[:, tq] == X[:, tq]
    Sm = lambda Z: torch.einsum('bki,bkj->bij', Z[:, :tq], Z[:, :tq]) / tq
    M = torch.einsum('bki,bkj->bij', X[:, 1:tq + 1], X[:, :tq]) / tq
    pred_dir = -torch.einsum('bij,bjk,bk->bi', M, Sm(Xd) - Sm(X), X[:, tq])
    with torch.no_grad(): m(Xd); zd = m.cache['z'][0]
    for gname, hs in [('similarity heads (L0, non-prev)', [h for h in range(4) if h != PREV]), ('prev-token head (control)', [PREV])]:
        def f(z, hs=hs): z = z.clone(); z[:, hs, tq] = zd[:, hs, tq]; return z
        with torch.no_grad(): Pp = m(X, {('z', 0): f})
        dP = Pp[:, tq] - P0[:, tq]; cs = F.cosine_similarity(dP, pred_dir, dim=-1)
        print(f'  t={tq:2d} patch {gname:32s} MSE vs LS {((P0[:, tq] - ls[:, tq]) ** 2).sum(-1).mean():.4f} -> {((Pp[:, tq] - ls[:, tq]) ** 2).sum(-1).mean():.4f}   cos(shift, -M(S\'-S)x_t) {cs.mean():+.3f}  ({(cs > 0).float().mean():.0%} positive)')

print('\n--- Test 3: replace layer-1 attention (queries t>=8, keys j>=1) with fitted idealised patterns ---')
cnt = torch.arange(1, T + 1, device=dev).float()[None, :, None, None]
def feats(X):
    S = torch.cumsum(torch.einsum('bti,btj->btij', X, X), 1) / cnt; Sx = torch.einsum('btij,btj->bti', S, X)
    return {'x': X, 'Sx': Sx}
specs = {'pure dot product <x_t, x_(j-1)>': ['x'], 'S-preconditioned (x, Sx) bilinear': ['x', 'Sx']}
msk = torch.zeros(T, T, dtype=torch.bool, device=dev); msk[8:, 1:] = True; msk &= ~m.mask
def bil(Fd, names):
    f = torch.cat([Fd[a] for a in names], -1); k = torch.cat([torch.zeros_like(f[:, :1]), f[:, :-1]], 1)   # key j carries token j-1
    return torch.einsum('bti,bjk->btjik', f, k).flatten(-2)
def centre(z, feat):
    mk = msk[None, :, :, None].float() if feat else msk[None].float(); dim = -2 if feat else -1
    return (z - (z * mk).sum(dim, keepdim=True) / mk.sum(dim, keepdim=True).clamp_min(1)) * mk
b1 = m.blocks[1]
def logits(x):
    q, k, _ = b1.qkv(b1.ln1(x)).view(x.shape[0], T, 3, b1.H, b1.dh).unbind(2); return torch.einsum('bihk,bjhk->bhij', q, k) / math.sqrt(b1.dh)
Xf, _, _ = make_batch(384)
with torch.no_grad(): m(Xf); lg = logits(m.cache['resid'][1])
Ff, Fx, Wfit = feats(Xf), feats(X), {}
for sp, names in specs.items():
    rows = centre(bil(Ff, names), True)[:, msk]; rows = rows.reshape(-1, rows.shape[-1])
    for h in range(4): Wfit[(sp, h)] = torch.linalg.lstsq(rows, centre(lg[:, h], False)[:, msk].reshape(-1, 1)).solution.squeeze(-1)
def replace(sp):
    Fz = bil(Fx, specs[sp])
    def f(att, x):
        att = att.clone()
        for h in range(4):
            mass = (att[:, h] * msk.float()).sum(-1, keepdim=True)
            new = (Fz @ Wfit[(sp, h)]).masked_fill(~msk[None], float('-inf')).softmax(-1).nan_to_num(0.) * mass
            att[:, h] = torch.where(msk[None], new, att[:, h])
        return att
    return f
e = lambda P: ((P - ls) ** 2).sum(-1)[:, 8:].mean().item()
print(f'  real attention:                     MSE vs LS {e(P0):.4f}')
for sp in specs:
    with torch.no_grad(): Pp = m(X, {('att', 1): replace(sp)})
    print(f'  {sp:35s} MSE vs LS {e(Pp):.4f}')
