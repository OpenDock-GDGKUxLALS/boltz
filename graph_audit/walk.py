"""Driver for the scripted pdb walk-through (see pdb_walk.cmds).

    python -m pdb walk.py <run_dir> < pdb_walk.cmds
"""

import sys

import torch

from harness import load_batch, load_model, pick_device, run_forward

run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/cli_prot/boltz_results_prot"
affinity = "--affinity" in sys.argv

torch.set_grad_enabled(False)
device = pick_device()
batch = load_batch(run_dir, device, affinity=affinity)
model = load_model(device, affinity=affinity, recycling_steps=1, sampling_steps=3,
                   diffusion_samples=2)
out = run_forward(model, batch)
print("OUTPUT KEYS:", sorted(out))
