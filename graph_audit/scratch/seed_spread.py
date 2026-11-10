"""Affinity value across seeds, unpadded vs padded crop (eager)."""
import json
import torch
from harness import load_model, pick_device, run_forward

dev = pick_device()
torch.set_grad_enabled(False)
model = load_model(dev, affinity=True)
feats = torch.load("scratch/link_feats.pt")
res = {}
for name in ("unpadded", "padded"):
    f = {k: v.to(dev) for k, v in feats[name].items()}
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        o = run_forward(model, f)
        res[f"{name}_seed{seed}"] = [o["affinity_pred_value"].item(), o["affinity_probability_binary"].item()]
        print(name, seed, res[f"{name}_seed{seed}"], flush=True)
json.dump(res, open("reports/affinity_seed_spread.json", "w"), indent=1)
