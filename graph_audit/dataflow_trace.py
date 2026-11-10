"""Record the tensor shapes that cross each module boundary in one real forward.

    python dataflow_trace.py --run runs/cli_prot/boltz_results_prot --samples 2
    python dataflow_trace.py --run runs/cli_aff/boltz_results_affinity --affinity
"""

from __future__ import annotations

import argparse
import inspect
from functools import partial

import torch

from harness import load_batch, load_model, pick_device, run_forward

STRUCT = [
    "input_embedder", "input_embedder.atom_encoder", "input_embedder.atom_attention_encoder",
    "rel_pos", "contact_conditioning", "template_module", "msa_module", "pairformer_module",
    "distogram_module", "diffusion_conditioning", "diffusion_conditioning.pairwise_conditioner",
    "diffusion_conditioning.atom_encoder", "structure_module.score_model",
    "structure_module.score_model.single_conditioner",
    "structure_module.score_model.atom_attention_encoder",
    "structure_module.score_model.token_transformer",
    "structure_module.score_model.atom_attention_decoder", "bfactor_module",
    "confidence_module", "confidence_module.pairformer_stack", "confidence_module.confidence_heads",
]
AFF = ["affinity_module1", "affinity_module1.pairformer_stack", "affinity_module1.affinity_heads"]


def desc(x, depth=0):
    if isinstance(x, torch.Tensor):
        return f"{tuple(x.shape)} {str(x.dtype).replace('torch.', '')}"
    if isinstance(x, partial):
        return f"partial({x.func.__name__}, indexing_matrix={tuple(x.keywords['indexing_matrix'].shape)})"
    if isinstance(x, dict):
        if "token_pad_mask" in x:
            return f"feats<{sum(isinstance(v, torch.Tensor) for v in x.values())} tensors>"
        if depth > 0:
            return "{...}"
        return "{" + ", ".join(f"{k}: {desc(v, depth + 1)}" for k, v in x.items()) + "}"
    if isinstance(x, (tuple, list)):
        return "(" + ", ".join(desc(v, depth + 1) for v in x) + ")"
    if x is None or isinstance(x, (int, float, bool)):
        return repr(x)
    return type(x).__name__


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--affinity", action="store_true")
    p.add_argument("--samples", type=int, default=None)
    args = p.parse_args()
    torch.set_grad_enabled(False)
    device = pick_device()
    batch = load_batch(args.run, device, affinity=args.affinity)
    model = load_model(device, affinity=args.affinity, recycling_steps=0, sampling_steps=2,
                       diffusion_samples=args.samples)
    seen = set()

    def hook(mod, a, kw, out, name):
        if name in seen:
            return
        seen.add(name)
        params = list(inspect.signature(mod.forward).parameters)
        named = {params[i]: v for i, v in enumerate(a) if i < len(params)}
        named.update(kw)
        ins = ", ".join(f"{k}={desc(v)}" for k, v in named.items()
                        if isinstance(v, (torch.Tensor, dict, partial)))
        print(f"\n[{name}]\n   in : {ins}\n   out: {desc(out)}")

    for name in (AFF if args.affinity else STRUCT):
        try:
            m = model.get_submodule(name)
        except AttributeError:
            continue
        m.register_forward_hook(partial(hook, name=name), with_kwargs=True)
    print("batch:", {k: tuple(batch[k].shape) for k in ["res_type", "msa", "ref_pos", "atom_to_token"]})
    out = run_forward(model, batch)
    print("\n[Boltz2.forward] out:", desc({k: v for k, v in out.items() if isinstance(v, torch.Tensor)}))


if __name__ == "__main__":
    main()
