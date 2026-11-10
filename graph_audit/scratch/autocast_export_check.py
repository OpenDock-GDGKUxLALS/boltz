"""Strict export on CPU under bf16 autocast (Boltz-2's bf16-mixed on CPU): the first
recycling pass must run outside the scan, or the scan carry changes dtype."""
import time

import torch

from export_full import Boltz2Program, count_nodes
from harness import load_model

dev = torch.device("cpu")
feats = {k: v.to(dev) for k, v in torch.load("reports/batch_boltz_results_prot.pt").items()}
model = load_model(dev, recycling_steps=1, sampling_steps=2, diffusion_samples=1)
model.requires_grad_(False)
program = Boltz2Program(model).eval()
with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
    torch.manual_seed(0)
    ref = program(feats)
    t0 = time.perf_counter()
    ep = torch.export.export(program, (feats,), strict=True)
    print(f"cpu autocast export: ok in {time.perf_counter() - t0:.0f} s, scans:",
          [k for k in count_nodes(ep.graph_module) if k.startswith("scan")])
    torch.manual_seed(0)
    got = ep.module()(feats)
print("cpu autocast exported vs eager max|d|:", max((got[k].float() - ref[k].float()).abs().max().item() for k in ref))
