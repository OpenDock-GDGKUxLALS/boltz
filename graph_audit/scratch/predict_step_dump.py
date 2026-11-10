"""Run predict_step on a saved batch and save the confidence outputs the writer reads."""
import sys

import torch
from harness import load_model, pick_device

dev = pick_device()
torch.set_grad_enabled(False)
batch = {k: v.to(dev) for k, v in torch.load(sys.argv[1]).items()}
model = load_model(dev, recycling_steps=0, sampling_steps=4, diffusion_samples=2)
torch.manual_seed(0)
out = model.predict_step(batch, 0)
pc = out["pair_chains_iptm"]
torch.save({"confidence_score": out["confidence_score"].cpu(), "ptm": out["ptm"].cpu(), "iptm": out["iptm"].cpu(),
            "pair_chains_iptm": {i: {j: pc[i][j].cpu() for j in pc[i]} for i in pc}}, sys.argv[2])
print("saved", sys.argv[2], "keys", sorted(pc), "score", out["confidence_score"].tolist())
