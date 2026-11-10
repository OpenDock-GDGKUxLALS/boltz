"""Export Boltz-2 structure inference as ONE torch.export program and check it.

The program covers the whole forward that `boltz predict` runs, minus file I/O:
input embedding -> recycling trunk (scan) -> diffusion conditioning -> 200-step
sampler (scan) -> confidence (pLDDT, PAE, PDE, pTM, ipTM, chain-pair ipTM) ->
ranking score. Inputs are the featurized tensors; outputs are tensors only.

Steering potentials are left out of the program: they only act when the input
has contact/pocket constraints or forced templates. `needs_steering()` checks
that on the host before running the program.

    PYTHONPATH=$PWD/wt-fixes/src python export_full.py --batch reports/batch_boltz_results_prot.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from harness import require_graph_fixes, load_model, pick_device, sync

require_graph_fixes()

from boltz.model.models.boltz2 import confidence_score  # noqa: E402

OUTPUTS = [
    "sample_atom_coords", "pdistogram", "plddt", "pae", "pde", "ptm", "iptm",
    "ligand_iptm", "protein_iptm", "pair_chains_iptm", "complex_plddt", "complex_iplddt",
    "complex_pde", "complex_ipde", "confidence_score",
]
NO_STEERING = {"fk_steering": False, "physical_guidance_update": False,
               "contact_guidance_update": False}


def needs_steering(feats: dict, steering_args: dict) -> bool:
    """Host-side version of the potential filter in AtomDiffusion.sample."""
    if steering_args["fk_steering"] or steering_args["physical_guidance_update"]:
        return True
    if not steering_args["contact_guidance_update"]:
        return False
    has_contacts = feats["contact_pair_index"].shape[-1] > 0
    has_forced_templates = "template_force" in feats and bool(feats["template_force"].any())
    return has_contacts or has_forced_templates


class Boltz2Program(nn.Module):
    """Boltz2.forward with the predict-time arguments fixed, tensors in and out."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        pa = model.predict_args
        self.kwargs = dict(
            recycling_steps=pa["recycling_steps"],
            num_sampling_steps=pa["sampling_steps"],
            diffusion_samples=pa["diffusion_samples"],
            max_parallel_samples=pa["max_parallel_samples"],
            run_confidence_sequentially=True,
        )
        # The program is for inputs without constraints (see needs_steering).
        model.steering_args = {**model.steering_args, **NO_STEERING}
        # Read the noise schedule now, outside tracing; the traced sampler
        # reuses the cached tensors as constants.
        sm = model.structure_module
        sm.step_schedule(pa["sampling_steps"], sm.step_scale)

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.model(feats, **self.kwargs)
        out["confidence_score"] = confidence_score(out["complex_plddt"], out["iptm"], out["ptm"])
        return {k: out[k] for k in OUTPUTS}


def count_nodes(gm) -> dict:
    """call_function nodes in the top graph and in each subgraph (scan bodies)."""
    counts = {"top": sum(1 for n in gm.graph.nodes if n.op == "call_function")}
    for name, sub in gm.named_children():
        if isinstance(sub, torch.fx.GraphModule):
            counts[name] = sum(1 for n in sub.graph.nodes if n.op == "call_function")
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True, help="saved feats from compare.py")
    p.add_argument("--recycles", type=int, default=3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/export_full.json")
    p.add_argument("--save", help="write the program to this .pt2 path")
    p.add_argument("--strict", action="store_true", help="trace with Dynamo (strict export)")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    device = pick_device()
    feats = {k: v.to(device) for k, v in torch.load(args.batch).items()}
    model = load_model(device, recycling_steps=args.recycles, sampling_steps=args.steps,
                       diffusion_samples=args.samples)
    model.requires_grad_(False)
    report = {"device": str(device), "torch": torch.__version__, "batch": args.batch,
              "strict": args.strict,
              "config": model.predict_args,
              "needs_steering": needs_steering(feats, model.steering_args)}
    assert not report["needs_steering"], "input has constraints: run it eagerly"
    program = Boltz2Program(model).eval()

    # Eager reference: the same module, not traced.
    torch.manual_seed(args.seed)
    sync(device)
    t0 = time.perf_counter()
    ref = program(feats)
    sync(device)
    report["eager_seconds"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    ep = torch.export.export(program, (feats,), strict=args.strict)
    report["export_seconds"] = round(time.perf_counter() - t0, 1)
    report["nodes"] = count_nodes(ep.graph_module)
    report["scans"] = sum(1 for n in ep.graph.nodes if n.op == "call_function"
                          and "scan" in str(n.target))
    print("exported:", report["export_seconds"], "s", report["nodes"], flush=True)

    run = ep.module()
    torch.manual_seed(args.seed)
    sync(device)
    t0 = time.perf_counter()
    got = run(feats)
    sync(device)
    report["exported_seconds"] = round(time.perf_counter() - t0, 2)
    report["max_abs_diff"] = {k: (got[k].float() - ref[k].float()).abs().max().item()
                              for k in OUTPUTS}
    report["values"] = {k: ref[k].flatten().tolist() for k in
                        ("ptm", "iptm", "confidence_score", "complex_plddt")}
    print(json.dumps(report["max_abs_diff"], indent=1))
    if args.save:
        torch.export.save(ep, args.save)
        report["saved"] = args.save
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
