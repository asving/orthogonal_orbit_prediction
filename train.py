"""python train.py L [--attnonly]  -> models/model_L{L}[_attnonly].pt   (12k steps, ~2-4 min on one H100)"""
import sys, math, torch, torch.nn as nn
from common import *
L, mlp = int(sys.argv[1]), '--attnonly' not in sys.argv
torch.manual_seed(0); m = TF(L, mlp).to(dev); steps, lr = 12000, 1e-3
opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
for s in range(steps):
    for g in opt.param_groups: g['lr'] = lr * min(1, s / 500) * (0.5 * (1 + math.cos(math.pi * s / steps)) * 0.99 + 0.01)
    X, Y, _ = make_batch(256); loss = ((m(X) - Y) ** 2).sum(-1).mean()
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if s % 2000 == 0: print(f'step {s} loss {loss.item():.4f}', flush=True)
torch.save(m.state_dict(), f'models/model_L{L}{"" if mlp else "_attnonly"}.pt')
