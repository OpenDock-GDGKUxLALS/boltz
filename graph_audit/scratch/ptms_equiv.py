"""compute_ptms: graph-fixes vs upstream on real feats/coords and random PAE logits.

Cases: as featurized; asym ids remapped above N (like an affinity crop of a many-chain
complex); and under CPU bf16 autocast (what Boltz-2's bf16-mixed does on CPU)."""
import contextlib
import importlib.util
import sys

import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


up, new = load(sys.argv[1], "up_cu"), load(sys.argv[2], "new_cu")
cases = [("mps", "featurized", None), ("cpu", "featurized", None), ("mps", "asym ids >= N", None),
         ("cpu", "cpu autocast", torch.bfloat16)]
for dev, label, autocast_dtype in cases:
    feats = {k: v.to(dev) for k, v in torch.load("reports/batch_boltz_results_affinity.pt").items()}
    if label == "asym ids >= N":
        feats["asym_id"] = torch.where(feats["asym_id"] == 1, 450, 7)  # N = 397
    coords = torch.load("reports/v2_ligstruct_orig.pt")["sample_atom_coords"].to(dev)
    m, n = coords.shape[0], feats["asym_id"].shape[1]
    logits = torch.randn(m, n, n, 64, generator=torch.Generator().manual_seed(0)).to(dev)
    ctx = torch.autocast("cpu", dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
    with ctx:
        a, b = up.compute_ptms(logits, coords, feats, m), new.compute_ptms(logits, coords, feats, m)
    exact = all(torch.equal(x, y) for x, y in zip(a[:4], b[:4]))
    ids = sorted(a[4])
    pair_up = torch.stack([torch.stack([a[4][i][j] for j in ids], -1) for i in ids], -2)
    pair_new = b[4][:, : len(ids), : len(ids)]
    print(f"{dev} {label:14s}: ptm/iptm/ligand/protein exact={exact}  ids={ids}  "
          f"chain-pair max|d|={(pair_up.float() - pair_new.float()).abs().max().item():.2e}")
