"""Check that the graph-fixes branch gives the same predictions as upstream.

    # run once per code version (PYTHONPATH selects the version)
    python compare.py save --run <dir> --out reports/out_orig.pt
    PYTHONPATH=wt-fixes/src python compare.py save --run <dir> --out reports/out_fixed.pt
    python compare.py diff reports/out_orig.pt reports/out_fixed.pt
"""

from __future__ import annotations

import argparse
import time

import torch

KEYS = ["sample_atom_coords", "plddt", "pae", "pde", "ptm", "iptm", "ligand_iptm",
        "protein_iptm", "pair_chains_iptm", "complex_plddt", "complex_iplddt", "complex_pde",
        "complex_ipde", "pdistogram", "s", "z", "affinity_pred_value",
        "affinity_probability_binary"]


def chain_pairs(out, asym_id):
    """pair_chains_iptm as a (samples, C, C) tensor over the asym ids present.

    Upstream returns a nested dict {c1: {c2: (samples,)}}; graph-fixes returns a
    (samples, N, N) tensor over chains in ascending asym id order.
    """
    pc = out.get("pair_chains_iptm")
    if isinstance(pc, dict):
        ids = sorted(pc)
        return torch.stack([torch.stack([pc[i][j] for j in ids], -1) for i in ids], -2)
    if isinstance(pc, torch.Tensor):
        n = len(torch.unique(asym_id))
        return pc[:, :n, :n]
    return None


def save(args):
    import boltz
    from harness import load_batch, load_model, pick_device, run_forward, sync

    torch.set_grad_enabled(False)
    device = pick_device()
    # Featurize once and reuse: featurization applies a random roto-translation
    # to reference conformers (torch global RNG), so two processes would
    # otherwise see different inputs.
    from pathlib import Path
    cache = Path(args.out).with_name(f"batch_{Path(args.run).name}{'_aff' if args.affinity else ''}.pt")
    if cache.exists():
        batch = {k: v.to(device) for k, v in torch.load(cache).items()}
    else:
        batch = load_batch(args.run, device, affinity=args.affinity)
        torch.save({k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}, cache)
        batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
    model = load_model(device, affinity=args.affinity, recycling_steps=args.recycles,
                       sampling_steps=args.steps, diffusion_samples=args.samples)
    torch.manual_seed(args.seed)
    sync(device)
    t0 = time.perf_counter()
    out = run_forward(model, batch)
    sync(device)
    secs = time.perf_counter() - t0
    out = dict(out)
    out["pair_chains_iptm"] = chain_pairs(out, batch["asym_id"])
    keep = {k: out[k].detach().float().cpu() for k in KEYS if isinstance(out.get(k), torch.Tensor)}
    keep["_meta"] = {"boltz_src": boltz.__file__, "seconds": secs, "config": model.predict_args}
    torch.save(keep, args.out)
    print(f"saved {args.out} from {boltz.__file__} in {secs:.1f} s")


def diff(args):
    a, b = torch.load(args.a), torch.load(args.b)
    print("A:", a["_meta"]["boltz_src"], f"{a['_meta']['seconds']:.1f} s")
    print("B:", b["_meta"]["boltz_src"], f"{b['_meta']['seconds']:.1f} s")
    for k in KEYS:
        if k not in a or k not in b:
            continue
        x, y = a[k], b[k]
        d = (x - y).abs()
        line = f"  {k:28s} shape={tuple(x.shape)!s:22s} max|d|={d.max().item():.3e}  mean|d|={d.mean().item():.3e}"
        if k == "sample_atom_coords":
            rmsd = ((x - y) ** 2).sum(-1).mean().sqrt().item()
            line += f"  RMSD={rmsd:.4f} A"
        print(line)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("--run", required=True)
    s.add_argument("--affinity", action="store_true")
    s.add_argument("--recycles", type=int, default=None)
    s.add_argument("--steps", type=int, default=200)
    s.add_argument("--samples", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    args = p.parse_args()
    save(args) if args.cmd == "save" else diff(args)


if __name__ == "__main__":
    main()
