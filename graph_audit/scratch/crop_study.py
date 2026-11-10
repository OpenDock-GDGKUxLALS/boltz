"""Reproduce the CLI affinity crop on the CPU and compare features:
CLI (re-tokenize + crop + featurize the predicted structure) vs gather from the
full-complex features of the structure stage."""
import torch
from pathlib import Path
from boltz.data.module.inferencev2 import load_input, PredictionDataset
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.types import Manifest

run = Path("runs/cli_aff/boltz_results_affinity")
manifest = Manifest.load(run / "processed/manifest.json")
record = [r for r in manifest.records if r.affinity][0]
inp = load_input(record, target_dir=run / "predictions", msa_dir=run / "processed/msa", affinity=True)
tok = Boltz2Tokenizer().tokenize(inp)
full_tokens = tok.tokens
crop = AffinityCropper().crop(tok, max_tokens=256, max_atoms=2048)
sel = torch.as_tensor(crop.tokens["token_idx"].astype("int64"))
print("full tokens", len(full_tokens), "cropped", len(sel), "atoms", int(crop.tokens["atom_num"].sum()))
torch.save({"token_idx": sel, "center_coords": torch.as_tensor(full_tokens["center_coords"]),
            "res_idx": torch.as_tensor(full_tokens["res_idx"].astype("int64")),
            "asym_id": torch.as_tensor(full_tokens["asym_id"].astype("int64")),
            "mol_type": torch.as_tensor(full_tokens["mol_type"].astype("int64")),
            "atom_num": torch.as_tensor(full_tokens["atom_num"].astype("int64")),
            "affinity_mask": torch.as_tensor(full_tokens["affinity_mask"]),
            "resolved_mask": torch.as_tensor(full_tokens["resolved_mask"])},
           "scratch/crop_ref.pt")

full = torch.load("reports/batch_boltz_results_affinity.pt")
aff = torch.load("reports/batch_boltz_results_affinity_aff.pt")
atom_tok = full["atom_to_token"][0].argmax(-1)  # token of each atom
real_atom = full["atom_pad_mask"][0].bool()
keep_atom = torch.isin(atom_tok, sel) & real_atom
atom_sel = torch.nonzero(keep_atom).flatten()
print("selected atoms", len(atom_sel), "aff atom pad", aff["atom_pad_mask"].shape[1], "real", int(aff["atom_pad_mask"].sum()))
torch.save({"token_idx": sel, "atom_idx": atom_sel}, "scratch/crop_sel.pt")
