"""New compute_frame_pred_inference vs upstream's loop compute_frame_pred(inference=True)."""
import importlib.util
import sys

import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


upstream = load(sys.argv[1], "upstream_confidence_utils")
new = load(sys.argv[2], "new_confidence_utils")
cases = [("protein+ligand", "reports/batch_boltz_results_affinity.pt", "reports/v2_ligstruct_orig.pt"),
         ("protein", "reports/batch_boltz_results_prot.pt", "reports/v2_prot_orig.pt")]
for dev in ("mps", "cpu"):
    for name, batch, out in cases:
        feats = {k: v.to(dev) for k, v in torch.load(batch).items()}
        coords = torch.load(out)["sample_atom_coords"].to(dev)  # (samples, M, 3) from upstream
        m = coords.shape[0]
        fa, ma = upstream.compute_frame_pred(coords, feats["frames_idx"], feats, m, inference=True)
        fb, mb = new.compute_frame_pred_inference(coords, feats["frames_idx"], feats, m)
        lig = int((feats["mol_type"] == 3).sum())
        changed = int((fa != feats["frames_idx"].repeat_interleave(m, 0).reshape(fa.shape)).any(-1).sum())
        print(f"{dev} {name:15s} ligand tokens={lig:3d} recomputed frames={changed:3d} "
              f"frames equal={torch.equal(fa, fb)} mask equal={torch.equal(ma, mb)}")
