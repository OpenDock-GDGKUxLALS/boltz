"""Proof of concept on the graph-fixes branch: Boltz-2 structure inference as
three torch.export programs and a thin Python loop.

  trunk   input embedding, templates, MSA module, Pairformer recycles, distogram
  cond    diffusion conditioning (atom encoder + per-layer attention biases)
  step    one reverse-diffusion step (score model + update), run 200 times

The script exports each piece from the loaded model, runs the pieces with
the same seed as an eager forward, and compares the sampled coordinates.

    PYTHONPATH=wt-fixes/src python export_poc.py --batch reports/batch_boltz_results_prot.pt
"""

from __future__ import annotations

import argparse
import json
import time
from math import sqrt
from pathlib import Path

import torch
from torch import nn

from harness import require_graph_fixes, load_model, pick_device, run_forward, sync

require_graph_fixes()


class Trunk(nn.Module):
    """Boltz2.forward up to the distogram (eval mode), recycles fixed at build time."""

    PARTS = ["input_embedder", "s_init", "z_init_1", "z_init_2", "rel_pos", "token_bonds",
             "token_bonds_type", "contact_conditioning", "s_norm", "z_norm", "s_recycle",
             "z_recycle", "template_module", "msa_module", "pairformer_module", "distogram_module"]

    def __init__(self, model, recycling_steps: int):
        super().__init__()
        # Hold only the submodules the trunk uses, so the exported program does not
        # carry the score model, confidence and other unused weights.
        for name in self.PARTS:
            if hasattr(model, name):
                setattr(self, name, getattr(model, name))
        self.bond_type_feature = model.bond_type_feature
        self.use_templates = model.use_templates
        self.use_kernels = model.use_kernels
        self.recycling_steps = recycling_steps

    def forward(self, feats: dict[str, torch.Tensor]):
        m = self
        s_inputs = m.input_embedder(feats)
        s_init = m.s_init(s_inputs)
        z_init = m.z_init_1(s_inputs)[:, :, None] + m.z_init_2(s_inputs)[:, None, :]
        relative_position_encoding = m.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + m.token_bonds(feats["token_bonds"].float())
        if m.bond_type_feature:
            z_init = z_init + m.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + m.contact_conditioning(feats)

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        for _ in range(self.recycling_steps + 1):
            s = s_init + m.s_recycle(m.s_norm(s))
            z = z_init + m.z_recycle(m.z_norm(z))
            if m.use_templates:
                z = z + m.template_module(z, feats, pair_mask, use_kernels=m.use_kernels)
            z = z + m.msa_module(z, s_inputs, feats, use_kernels=m.use_kernels)
            s, z = m.pairformer_module(s, z, mask=mask, pair_mask=pair_mask,
                                       use_kernels=m.use_kernels)
        pdistogram = m.distogram_module(z)
        return s_inputs, s, z, relative_position_encoding, pdistogram


class Cond(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.diffusion_conditioning = model.diffusion_conditioning

    def forward(self, s, z, relative_position_encoding, feats: dict[str, torch.Tensor]):
        return self.diffusion_conditioning(
            s_trunk=s, z_trunk=z,
            relative_position_encoding=relative_position_encoding, feats=feats,
        )


class Step(nn.Module):
    """AtomDiffusion.denoise_step with every input passed as a tensor."""

    def __init__(self, model):
        super().__init__()
        self.structure_module = model.structure_module

    def forward(self, atom_coords, scalars, atom_mask, s_inputs, s_trunk,
                feats: dict[str, torch.Tensor], q, c, keys_indexing_matrix,
                atom_enc_bias, atom_dec_bias, token_trans_bias):
        conditioning = {
            "q": q, "c": c, "keys_indexing_matrix": keys_indexing_matrix,
            "atom_enc_bias": atom_enc_bias, "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }
        cond = {"s_inputs": s_inputs, "s_trunk": s_trunk, "feats": feats,
                "diffusion_conditioning": conditioning}
        return self.structure_module.denoise_step(
            atom_mask, [atom_coords.shape[0]], cond, atom_coords, scalars)


def schedule_scalars(sm, num_steps, device):
    """Same per-step scalars AtomDiffusion.sample builds (host float64 math)."""
    sigmas = sm.sample_schedule(num_steps)
    gammas = torch.where(sigmas > sm.gamma_min, sm.gamma_0, 0.0)
    sh, gh = sigmas.tolist(), gammas.tolist()
    rows = []
    for sigma_tm, sigma_t, gamma in zip(sh[:-1], sh[1:], gh[1:]):
        t_hat = sigma_tm * (1 + gamma)
        noise_var = sm.noise_scale**2 * (t_hat**2 - sigma_tm**2)
        rows.append((t_hat, sqrt(noise_var), sm.step_scale * (sigma_t - t_hat)))
    return sigmas, torch.tensor(rows, dtype=torch.float32, device=device)


def count_nodes(ep):
    return sum(1 for n in ep.graph.nodes if n.op == "call_function")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--recycles", type=int, default=3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--out", default="reports/export_poc.json")
    p.add_argument("--save", action="store_true", help="write reports/{trunk,cond,step}.pt2")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    device = pick_device()
    feats = {k: v.to(device) for k, v in torch.load(args.batch).items()}
    model = load_model(device, recycling_steps=args.recycles, sampling_steps=args.steps,
                       diffusion_samples=args.samples)
    model.requires_grad_(False)
    sm = model.structure_module
    report = {"device": str(device), "torch": torch.__version__, "config": model.predict_args}

    # ---- reference: the eager model, same seed ---------------------------------
    torch.manual_seed(0)
    ref = run_forward(model, feats)

    # ---- example inputs for each piece (from one eager pass) -------------------
    trunk, cond, step = Trunk(model, args.recycles), Cond(model), Step(model)
    torch.manual_seed(0)
    s_inputs, s, z, relpos, _ = trunk(feats)
    cond_out = cond(s, z, relpos, feats)
    _, scalars = schedule_scalars(sm, args.steps, device)
    atom_mask = feats["atom_pad_mask"].float().repeat_interleave(args.samples, 0)
    coords0 = torch.randn((*atom_mask.shape, 3), device=device)
    step_args = (coords0, scalars[0], atom_mask, s_inputs.float(), s.float(), feats, *cond_out)

    # ---- export -----------------------------------------------------------------
    eps = {}
    for name, mod, ex_args in [
        ("cond", cond, (s, z, relpos, feats)),
        ("step", step, step_args),
        ("trunk", trunk, (feats,)),
    ]:
        t0 = time.perf_counter()
        try:
            eps[name] = torch.export.export(mod, ex_args, strict=False)
            report[name] = {"ok": True, "export_seconds": round(time.perf_counter() - t0, 1),
                            "call_function_nodes": count_nodes(eps[name])}
        except Exception as e:  # noqa: BLE001
            report[name] = {"ok": False, "export_seconds": round(time.perf_counter() - t0, 1),
                            "error": f"{type(e).__name__}: {str(e)[:1500]}"}
        print(name, report[name], flush=True)

    # ---- run the exported pieces end to end, same seed ---------------------------
    if all(report[k]["ok"] for k in ("trunk", "cond", "step")):
        trunk_ep, cond_ep, step_ep = (eps[k].module() for k in ("trunk", "cond", "step"))
        torch.manual_seed(0)
        sync(device)
        t0 = time.perf_counter()
        s_inputs, s, z, relpos, _ = trunk_ep(feats)
        cond_out = cond_ep(s, z, relpos, feats)
        sigmas, scalars = schedule_scalars(sm, args.steps, device)
        coords = sigmas[0] * torch.randn((*atom_mask.shape, 3), device=device)
        for i in range(args.steps):
            coords = step_ep(coords, scalars[i], atom_mask, s_inputs.float(), s.float(), feats, *cond_out)
        sync(device)
        report["exported_pipeline_seconds"] = round(time.perf_counter() - t0, 2)
        d = (coords - ref["sample_atom_coords"]).abs()
        report["max_abs_diff_vs_eager_coords"] = d.max().item()
        report["max_abs_diff_vs_eager_s"] = (s - ref["s"]).abs().max().item()
        report["max_abs_diff_vs_eager_z"] = (z - ref["z"]).abs().max().item()
        print("exported pipeline vs eager: coords max|d| =", report["max_abs_diff_vs_eager_coords"],
              " s =", report["max_abs_diff_vs_eager_s"], " z =", report["max_abs_diff_vs_eager_z"])
        if args.save:
            for k in ("trunk", "cond", "step"):
                torch.export.save(eps[k], f"reports/{k}.pt2")

    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
