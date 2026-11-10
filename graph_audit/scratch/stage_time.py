"""Per-stage wall time for one forward, twice in one process (first call vs warm)."""
import sys
import time
import torch
from harness import StageTimer, load_model, pick_device, run_forward, sync

dev = pick_device()
batch = {k: v.to(dev) for k, v in torch.load(sys.argv[1]).items()}
torch.set_grad_enabled(False)
model = load_model(dev, recycling_steps=int(sys.argv[2]), sampling_steps=int(sys.argv[3]), diffusion_samples=2)
names = ["input_embedder", "msa_module", "pairformer_module", "diffusion_conditioning",
         "structure_module", "confidence_module"]
for call in range(2):
    timer = StageTimer(model, dev, names)
    sync(dev); t0 = time.perf_counter()
    torch.manual_seed(0)
    run_forward(model, batch)
    sync(dev)
    total = time.perf_counter() - t0
    timer.remove()
    print(f"call {call}: total {total:.2f} s  " + "  ".join(f"{k}={timer.totals[k]:.2f}" for k in names))
