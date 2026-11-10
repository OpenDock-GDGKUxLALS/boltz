"""affinity_link vs the CLI path (tokenize + AffinityCropper + featurizer), on CPU."""
from pathlib import Path

import numpy as np
import torch

from affinity_link import center_coords, crop_and_gather, crop_mask
from boltz.data import const
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.module.inferencev2 import load_input
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import Manifest

run = Path("runs/cli_aff/boltz_results_affinity")
full = torch.load("reports/batch_boltz_results_affinity.pt")
cli = torch.load("reports/batch_boltz_results_affinity_aff.pt")

# --- the CLI's crop of the predicted structure -------------------------------------
manifest = Manifest.load(run / "processed/manifest.json")
record = [r for r in manifest.records if r.affinity][0]
inp = load_input(record, target_dir=run / "predictions", msa_dir=run / "processed/msa", extra_mols_dir=run / "processed/mols", affinity=True)
tokenized = Boltz2Tokenizer().tokenize(inp)
tokens = tokenized.tokens
cropped = AffinityCropper().crop(tokenized, max_tokens=256, max_atoms=2048)
cli_sel = torch.as_tensor(np.sort(cropped.tokens["token_idx"]).astype("int64"))

# Predicted atom coordinates in feature atom order (tokens' atom ranges, in order).
atoms = inp.structure.atoms
idx = np.concatenate([np.arange(t["atom_idx"], t["atom_idx"] + t["atom_num"]) for t in tokens])
M = full["atom_pad_mask"].shape[1]
coords = torch.zeros(M, 3)
coords[: len(idx)] = torch.as_tensor(atoms["coords"][idx])
cc = center_coords(full, coords)
print("center coords == tokenizer center_coords:", torch.equal(cc, torch.as_tensor(tokens["center_coords"])))

mask = crop_mask(full, cc)
sel = torch.nonzero(mask).flatten()
print(f"crop: tensor {len(sel)} tokens, CLI {len(cli_sel)} tokens, identical: {torch.equal(sel, cli_sel)}")

# --- unpadded gather vs the CLI's affinity features ---------------------------------
lig_rows = full["mol_type"][0, sel] == const.chain_type_ids["NONPOLYMER"]


def compare(name, got, ref, skip=("ref_pos",)):
    bad = []
    for k, y in ref.items():
        if k not in got or k in skip:
            continue
        x = got[k]
        if k == "frames_idx":  # ligand frames are recomputed from coordinates in confidence
            x, y = x[:, :, : len(sel)][:, :, ~lig_rows], y[:, :, : len(sel)][:, :, ~lig_rows]
        if x.shape != y.shape or not torch.equal(x.to(y.dtype), y):
            bad.append((k, tuple(x.shape), tuple(y.shape)))
    print(f"{name}: {len(got)} features built, {len(bad)} differ{': ' + str(bad) if bad else ''}")


got = crop_and_gather(full, mask)
compare("unpadded vs CLI featurizer", got, {k: cli[k] for k in got})

# --- padded gather vs the featurizer run with the same padding -----------------------
feat = Boltz2Featurizer()
mols = {}
mols.update(load_canonicals(Path("~/.boltz/mols").expanduser()))
mols.update(inp.extra_mols)
names = set(cropped.tokens["res_name"].tolist()) - set(mols)
mols.update(load_molecules(Path("~/.boltz/mols").expanduser(), names))
ref_pad = feat.process(cropped, molecules=mols, random=np.random.default_rng(42), training=False,
                       max_atoms=2048, max_tokens=256, max_seqs=const.max_msa_seqs,
                       pad_to_max_seqs=False, single_sequence_prop=0.0, compute_frames=True,
                       inference_pocket_constraints=None, inference_contact_constraints=None,
                       compute_constraint_features=True, override_method="other",
                       compute_affinity=True)
ref_pad = {k: v[None] for k, v in ref_pad.items() if isinstance(v, torch.Tensor)}
got_pad = crop_and_gather(full, mask, pad_tokens=256, pad_atoms=2048)
compare("padded (256/2048) vs featurizer padded", got_pad, {k: ref_pad[k] for k in got_pad})
torch.save({"unpadded": got, "padded": got_pad, "sel": sel}, "scratch/link_feats.pt")
