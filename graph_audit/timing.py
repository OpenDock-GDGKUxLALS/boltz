"""Steady-state eager timing of one prediction (after a warm-up run).

    python timing.py --batch reports/batch_boltz_results_prot.pt [--repeat 3]
    PYTHONPATH=wt-fixes/src python timing.py --batch ...
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

import boltz
from harness import load_model, pick_device, run_forward, sync


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--affinity", action="store_true")
    p.add_argument("--recycles", type=int, default=None)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    torch.set_grad_enabled(False)
    device = pick_device()
    batch = {k: v.to(device) for k, v in torch.load(args.batch).items()}
    model = load_model(device, affinity=args.affinity, recycling_steps=args.recycles,
                       sampling_steps=args.steps, diffusion_samples=args.samples)
    torch.manual_seed(0)
    run_forward(model, batch)  # warm-up
    times = []
    for _ in range(args.repeat):
        torch.manual_seed(0)
        sync(device)
        t0 = time.perf_counter()
        run_forward(model, batch)
        sync(device)
        times.append(time.perf_counter() - t0)
    print(f"{boltz.__file__}: {device} config={model.predict_args}")
    print(f"  times: {[round(t, 2) for t in times]}  median={statistics.median(times):.2f} s")


if __name__ == "__main__":
    main()
