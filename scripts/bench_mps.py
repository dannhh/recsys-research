"""Measure training throughput of a TIGER-size encoder-decoder on this Mac (MPS)."""
import time
import torch
import torch.nn as nn

dev = torch.device("mps")
V, D, L_ENC, L_DEC = 1024 + 2000 + 4, 128, 4, 4  # SID tokens + hashed user tokens + specials; TIGER sizes
model = nn.Transformer(d_model=D, nhead=6 if D % 6 == 0 else 4, num_encoder_layers=L_ENC, num_decoder_layers=L_DEC,
                       dim_feedforward=1024, dropout=0.1, batch_first=True).to(dev)
emb, head = nn.Embedding(V, D).to(dev), nn.Linear(D, V).to(dev)
params = sum(p.numel() for m in (model, emb, head) for p in m.parameters())
opt = torch.optim.AdamW(list(model.parameters()) + list(emb.parameters()) + list(head.parameters()), lr=1e-3)
print(f"params {params/1e6:.1f}M")
for ctx_items in (20, 50, 100):
    B, S, T = 256, ctx_items * 4, 5
    src = torch.randint(0, V, (B, S), device=dev); tgt = torch.randint(0, V, (B, T), device=dev)
    for i in range(13):
        if i == 3:
            torch.mps.synchronize(); t0 = time.time()
        out = model(emb(src), emb(tgt[:, :-1]), tgt_mask=model.generate_square_subsequent_mask(T - 1, device=dev))
        loss = nn.functional.cross_entropy(head(out).reshape(-1, V), tgt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.mps.synchronize(); dt = (time.time() - t0) / 10
    print(f"context {ctx_items} items ({S} tokens): {dt*1000:.0f} ms/step, {B/dt:,.0f} examples/s")
