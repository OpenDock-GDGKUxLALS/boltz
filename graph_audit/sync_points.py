"""Count device->host synchronizations during one eager prediction, by call site.

A sync happens whenever Python needs a value that lives on the GPU:
Tensor.item(), bool(tensor) in an `if`, .tolist(), int()/float(), .cpu(), ...
Each one stalls the device queue and is also a place where a captured graph
(CUDA graph, exported program) cannot continue.

    python sync_points.py --run runs/cli_prot/boltz_results_prot --out reports/sync_prot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from harness import load_batch, load_model, pick_device, run_forward

import re
SRC_RE = re.compile(r".*/src/(?=boltz/)")
HITS = Counter()
ENABLED = [False]


def call_site() -> str:
    f = sys._getframe(2)
    while f is not None:
        if SRC_RE.match(f.f_code.co_filename) and "site-packages" not in f.f_code.co_filename:
            return f"{SRC_RE.sub('', f.f_code.co_filename)}:{f.f_lineno} ({f.f_code.co_name})"
        f = f.f_back
    return "?"


def patch(name: str):
    orig = getattr(torch.Tensor, name)

    def wrapper(self, *a, **kw):
        if ENABLED[0] and self.device.type != "cpu":
            HITS[(name, call_site())] += 1
        return orig(self, *a, **kw)
    setattr(torch.Tensor, name, wrapper)


for n in ["item", "tolist", "__bool__", "__int__", "__float__", "__index__", "cpu", "numpy"]:
    patch(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--affinity", action="store_true")
    p.add_argument("--recycles", type=int, default=None)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_grad_enabled(False)
    device = pick_device()
    batch = load_batch(args.run, device, affinity=args.affinity)
    model = load_model(device, affinity=args.affinity, recycling_steps=args.recycles,
                       sampling_steps=args.steps, diffusion_samples=args.samples)
    ENABLED[0] = True
    run_forward(model, batch)
    ENABLED[0] = False
    rows = [{"kind": k, "site": s, "count": n} for (k, s), n in HITS.most_common()]
    total = sum(HITS.values())
    Path(args.out).write_text(json.dumps({"config": model.predict_args, "total": total, "sites": rows}, indent=1, default=str))
    print(f"config={model.predict_args}")
    print(f"TOTAL device->host syncs in one forward: {total}")
    for r in rows:
        print(f"  {r['count']:6d}  {r['kind']:10s} {r['site']}")


if __name__ == "__main__":
    main()
