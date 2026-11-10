"""Where does eager Boltz-2 inference spend its time?

    python profile_stages.py --run runs/cli_prot/boltz_results_prot [--affinity]

Every timed region synchronizes the device before and after, so the numbers
are attributable but slightly pessimistic (no async overlap across regions).
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

import boltz.model.layers.confidence_utils as conf_utils
import boltz.model.modules.confidencev2 as confv2
import boltz.model.modules.diffusionv2 as diffv2
from boltz.model.potentials.potentials import Potential
from harness import StageTimer, load_batch, load_model, pick_device, run_forward, sync

TOTALS = defaultdict(float)
CALLS = defaultdict(int)


def timed_fn(label, fn, device):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        sync(device)
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        sync(device)
        TOTALS[label] += time.perf_counter() - t0
        CALLS[label] += 1
        return out
    return wrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--affinity", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--recycles", type=int, default=None)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.set_grad_enabled(False)
    device = pick_device(args.device)
    batch = load_batch(args.run, device, affinity=args.affinity)
    model = load_model(device, affinity=args.affinity, recycling_steps=args.recycles,
                       sampling_steps=args.steps, diffusion_samples=args.samples)

    # warm-up (MPS/CUDA kernel caches), short config
    pa = dict(model.predict_args)
    model.predict_args.update(recycling_steps=0, sampling_steps=2)
    run_forward(model, batch)
    model.predict_args.update(pa)

    names = ["input_embedder", "rel_pos", "contact_conditioning", "msa_module",
             "pairformer_module", "distogram_module", "diffusion_conditioning",
             "structure_module.score_model", "confidence_module"]
    if hasattr(model, "template_module"):
        names.insert(3, "template_module")
    if hasattr(model, "bfactor_module"):
        names.append("bfactor_module")
    if args.affinity:
        names += ["affinity_module1", "affinity_module2"]
    timer = StageTimer(model, device, names)

    # Function-level regions inside the sampler and the confidence heads.
    sm = model.structure_module
    sm.sample = timed_fn("structure_module.sample (whole loop)", sm.sample, device)
    diffv2.weighted_rigid_align = timed_fn("  weighted_rigid_align", diffv2.weighted_rigid_align, device)
    diffv2.compute_random_augmentation = timed_fn("  compute_random_augmentation", diffv2.compute_random_augmentation, device)
    Potential.compute_gradient = timed_fn("  potentials.compute_gradient", Potential.compute_gradient, device)
    confv2.compute_ptms = timed_fn("  confidence.compute_ptms", confv2.compute_ptms, device)
    conf_utils.compute_frame_pred = timed_fn("  confidence.compute_frame_pred", conf_utils.compute_frame_pred, device)

    sync(device)
    t0 = time.perf_counter()
    out = run_forward(model, batch)
    sync(device)
    total = time.perf_counter() - t0
    timer.remove()

    rows = []
    for k in names:
        rows.append({"region": k, "seconds": timer.totals[k], "calls": timer.calls[k]})
    for k in TOTALS:
        rows.append({"region": k, "seconds": TOTALS[k], "calls": CALLS[k]})
    res = {
        "device": str(device),
        "config": model.predict_args,
        "n_tokens": int(batch["token_pad_mask"].shape[1]),
        "n_atoms": int(batch["atom_pad_mask"].shape[1]),
        "n_msa": int(batch["msa"].shape[1]),
        "total_seconds": total,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"device={device} tokens={res['n_tokens']} atoms={res['n_atoms']} msa={res['n_msa']} "
          f"config={model.predict_args}")
    print(f"TOTAL forward: {total:.2f} s")
    for r in rows:
        print(f"  {r['region']:42s} {r['seconds']:8.2f} s  {100 * r['seconds'] / total:5.1f}%  calls={r['calls']}")


if __name__ == "__main__":
    main()
