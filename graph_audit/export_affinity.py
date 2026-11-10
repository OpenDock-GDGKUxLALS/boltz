"""Structure model -> affinity model on the device, exported as ONE program.

The CLI writes the top-ranked structure to pre_affinity_<id>.npz, then
re-tokenizes, crops and featurizes it on the CPU before running the affinity
model. Here that hand-off is tensor code (affinity_link.py), so
feats -> structure model -> crop -> affinity model is one graph.

Checks:
  link    eager affinity model on the gathered crop (CLI's random conformer
          rotation substituted) vs on the CLI's own features: should be identical.
  padding eager: crop padded to 256 tokens / 2048 atoms (static shapes) vs unpadded.
  export  the fused program, exported, vs the same module run eagerly.

    PYTHONPATH=$PWD/wt-fixes/src python export_affinity.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from affinity_link import center_coords, crop_and_gather, crop_mask
from export_full import Boltz2Program, count_nodes, needs_steering
from harness import require_graph_fixes, load_model, pick_device, run_forward, sync

require_graph_fixes()

AFF_OUT = ["affinity_pred_value", "affinity_probability_binary", "affinity_pred_value1",
           "affinity_probability_binary1", "affinity_pred_value2", "affinity_probability_binary2"]


class AffinityProgram(nn.Module):
    """Crop + gather + affinity model (Boltz2 with the affinity weights)."""

    def __init__(self, model, pad_tokens=256, pad_atoms=2048):
        super().__init__()
        self.model = model
        self.pad_tokens, self.pad_atoms = pad_tokens, pad_atoms
        pa = model.predict_args
        self.kwargs = dict(
            recycling_steps=pa["recycling_steps"],
            num_sampling_steps=pa["sampling_steps"],
            diffusion_samples=pa["diffusion_samples"],
            max_parallel_samples=pa["max_parallel_samples"],
            run_confidence_sequentially=True,
        )
        sm = model.structure_module
        sm.step_schedule(pa["sampling_steps"], sm.step_scale)

    def forward(self, feats, atom_coords):
        """feats: structure-stage features; atom_coords: (M, 3) chosen sample."""
        keep = crop_mask(feats, center_coords(feats, atom_coords))
        aff_feats = crop_and_gather(feats, keep, self.pad_tokens, self.pad_atoms)
        out = self.model(aff_feats, **self.kwargs)
        return {k: out[k] for k in AFF_OUT if k in out}


class FullProgram(nn.Module):
    """feats -> structure + confidence -> top-ranked sample -> affinity."""

    def __init__(self, structure: Boltz2Program, affinity: AffinityProgram):
        super().__init__()
        self.structure = structure
        self.affinity = affinity

    def forward(self, feats):
        out = self.structure(feats)
        # The writer ranks samples by confidence_score; rank 0 goes to affinity.
        best = torch.argmax(out["confidence_score"])
        coords = out["sample_atom_coords"].index_select(0, best[None])[0]
        return {**out, **self.affinity(feats, coords)}


def diff(a, b, keys):
    return {k: (a[k].float() - b[k].float()).abs().max().item() for k in keys if k in a and k in b}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", default="reports/batch_boltz_results_affinity.pt",
                   help="structure-stage feats of the protein-ligand input")
    p.add_argument("--cli-aff", default="reports/batch_boltz_results_affinity_aff.pt",
                   help="affinity feats the CLI built (crop of its top-ranked structure)")
    p.add_argument("--link-feats", default="scratch/link_feats.pt",
                   help="crop_and_gather of the same crop (scratch/link_check.py)")
    p.add_argument("--cli-out", default="reports/v2_aff_fixed.pt",
                   help="compare.py output of the affinity model on the CLI feats, seed 0")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--recycles", type=int, default=3, help="structure model")
    p.add_argument("--samples", type=int, default=2, help="structure model")
    p.add_argument("--checks", default="link,padding,export")
    p.add_argument("--out", default="reports/export_affinity.json")
    args = p.parse_args()
    checks = args.checks.split(",")

    torch.set_grad_enabled(False)
    device = pick_device()
    report = {"device": str(device), "torch": torch.__version__}
    aff_model = load_model(device, affinity=True, sampling_steps=args.steps)
    aff_model.requires_grad_(False)
    report["affinity_config"] = aff_model.predict_args

    if "link" in checks:
        # Same crop, CLI path vs gather path, through the eager affinity model.
        cli = {k: v.to(device) for k, v in torch.load(args.cli_aff).items()}
        link = {k: v.to(device) for k, v in torch.load(args.link_feats)["unpadded"].items()}
        link["ref_pos"] = cli["ref_pos"]  # the featurizer's own random conformer rotation
        torch.manual_seed(0)
        got = run_forward(aff_model, link)
        ref = torch.load(args.cli_out)
        report["link_vs_cli"] = diff({k: v.cpu() for k, v in got.items() if isinstance(v, torch.Tensor)},
                                     ref, AFF_OUT + ["sample_atom_coords", "z", "iptm"])
        print("link vs CLI:", report["link_vs_cli"], flush=True)

    if "padding" in checks:
        feats = torch.load(args.link_feats)
        res = {}
        for name in ("unpadded", "padded"):
            f = {k: v.to(device) for k, v in feats[name].items()}
            torch.manual_seed(0)
            o = run_forward(aff_model, f)
            n = int(feats["unpadded"]["token_pad_mask"].sum())
            res[name] = {k: o[k].flatten().tolist() for k in AFF_OUT if k in o}
            res[name]["z_real"] = o["z"][:, :n, :n].cpu()
        report["padding"] = {k: (res["unpadded"][k], res["padded"][k]) for k in AFF_OUT if k in res["padded"]}
        report["padding"]["z_real_max_abs_diff"] = (res["unpadded"]["z_real"] - res["padded"]["z_real"]).abs().max().item()
        print("padding:", json.dumps(report["padding"], indent=1), flush=True)

    if "export" in checks:
        full = {k: v.to(device) for k, v in torch.load(args.full).items()}
        struct_model = load_model(device, recycling_steps=args.recycles, sampling_steps=args.steps,
                                  diffusion_samples=args.samples)
        struct_model.requires_grad_(False)
        report["structure_config"] = struct_model.predict_args
        assert not needs_steering(full, struct_model.steering_args)
        program = FullProgram(Boltz2Program(struct_model), AffinityProgram(aff_model)).eval()

        torch.manual_seed(0)
        sync(device)
        t0 = time.perf_counter()
        ref = program(full)
        sync(device)
        report["eager_seconds"] = round(time.perf_counter() - t0, 1)

        t0 = time.perf_counter()
        ep = torch.export.export(program, (full,), strict=True)
        report["export_seconds"] = round(time.perf_counter() - t0, 1)
        report["nodes"] = count_nodes(ep.graph_module)
        print("exported:", report["export_seconds"], "s", flush=True)

        torch.manual_seed(0)
        sync(device)
        t0 = time.perf_counter()
        got = ep.module()(full)
        sync(device)
        report["exported_seconds"] = round(time.perf_counter() - t0, 1)
        report["export_max_abs_diff"] = diff(got, ref, list(ref))
        report["values"] = {k: ref[k].flatten().tolist() for k in AFF_OUT + ["confidence_score", "iptm"]
                            if k in ref}
        print("exported vs eager:", json.dumps(report["export_max_abs_diff"], indent=1))

    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
