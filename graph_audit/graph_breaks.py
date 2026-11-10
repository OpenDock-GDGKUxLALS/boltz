"""Find where Dynamo / torch.export cannot keep Boltz-2 inference in one graph.

Subcommands (all take --run <processed run dir> and write JSON to --out):

  explain   torch._dynamo.explain over the full Boltz2.forward. Lists every
            graph break with its user stack (deduplicated by source line).
  count     torch.compile(model.forward) with a counting backend: how many
            graphs get compiled and how many graph launches one prediction
            makes, plus every recompile.
  modules   For each top-level submodule: capture its real first-call inputs,
            then run dynamo explain on it alone and try torch.export
            (non-strict and strict).
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import logging
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch._dynamo

from harness import load_batch, load_model, pick_device, run_forward, sync

import re
SRC_RE = re.compile(r"[^\"]*/src/(?=boltz/)")


def is_user(path: str) -> bool:
    return bool(SRC_RE.match(path)) and "site-packages" not in path


def short(path: str) -> str:
    return SRC_RE.sub("", path).replace(str(Path(torch.__file__).parent.parent) + "/", "")


def frame_str(f) -> str:
    return f"{short(f.filename)}:{f.lineno} ({f.name})"


def tensor_only(feats: dict) -> dict:
    return {k: v for k, v in feats.items() if isinstance(v, torch.Tensor)}


# ----------------------------------------------------------------------------
# explain
# ----------------------------------------------------------------------------
def _user_line(stack_trace: str | None) -> str:
    """Innermost boltz source line from an FX node's stack_trace."""
    if not stack_trace:
        return "?"
    lines = [l.strip() for l in stack_trace.strip().splitlines() if l.strip().startswith("File")]
    user = [l for l in lines if is_user(l.split('"')[1] if '"' in l else l)]
    if not user:
        return "?"
    l = user[-1]
    # File "/.../x.py", line 12, in fn
    try:
        path = l.split('"')[1]
        lineno = l.split("line ")[1].split(",")[0]
        fn = l.split(" in ")[-1]
        return f"{short(path)}:{lineno} ({fn})"
    except Exception:  # noqa: BLE001
        return l


def graph_map(ex) -> list[dict]:
    rows = []
    for i, gm in enumerate(ex.graphs):
        ops = [n for n in gm.graph.nodes if n.op in ("call_function", "call_method", "call_module")]
        if not ops:
            rows.append({"graph": i, "ops": 0})
            continue
        rows.append({
            "graph": i,
            "ops": len(ops),
            "starts_at": _user_line(ops[0].meta.get("stack_trace")),
            "ends_at": _user_line(ops[-1].meta.get("stack_trace")),
        })
    return rows


def summarize_explain(ex) -> dict:
    sites = {}
    for br in ex.break_reasons:
        stack = [f for f in br.user_stack if "site-packages/torch" not in f.filename]
        top = stack[-1] if stack else (br.user_stack[-1] if br.user_stack else None)
        key = frame_str(top) if top else "?"
        reason = " ".join(br.reason.split())[:400]
        site = sites.setdefault(key, {"where": key, "reasons": Counter(), "count": 0,
                                       "stack": [frame_str(f) for f in stack[-8:]]})
        site["count"] += 1
        site["reasons"][reason] += 1
    out = {
        "graph_count": ex.graph_count,
        "graph_break_count": ex.graph_break_count,
        "op_count": ex.op_count,
        "ops_per_graph": [len(o) for o in ex.ops_per_graph],
        "break_sites": [
            {**s, "reasons": [{"reason": r, "n": n} for r, n in s["reasons"].most_common()]}
            for s in sites.values()
        ],
        "breaks_in_order": [
            {"reason": " ".join(br.reason.split())[:600],
             "user_stack": [frame_str(f) for f in br.user_stack if is_user(f.filename)]}
            for br in ex.break_reasons
        ],
        "graph_map": graph_map(ex),
    }
    return out


def cmd_explain(args, device):
    batch = load_batch(args.run, device, affinity=args.affinity)
    model = load_model(device, affinity=args.affinity, recycling_steps=args.recycles,
                       sampling_steps=args.steps, diffusion_samples=args.samples)
    if getattr(args, "sampler_outside", False):
        # Keep the 200-step loop in Python; Dynamo still compiles what it calls
        # (the per-step function) as its own frame.
        sm = model.structure_module
        sm.sample = torch.compiler.disable(sm.sample, recursive=False)
    torch._dynamo.reset()
    pa = model.predict_args
    t0 = time.perf_counter()
    with torch.no_grad():
        ex = torch._dynamo.explain(model.forward)(
            batch,
            recycling_steps=pa["recycling_steps"],
            num_sampling_steps=pa["sampling_steps"],
            diffusion_samples=pa["diffusion_samples"],
            max_parallel_samples=pa["max_parallel_samples"],
            run_confidence_sequentially=True,
        )
    res = summarize_explain(ex)
    res["seconds"] = time.perf_counter() - t0
    res["config"] = pa
    return res


# ----------------------------------------------------------------------------
# count
# ----------------------------------------------------------------------------
class CountingBackend:
    """Eager backend that records every compiled graph and every launch."""

    def __init__(self):
        self.graphs = []  # (n_ops, first user frame)
        self.launches = Counter()

    def __call__(self, gm, example_inputs):
        idx = len(self.graphs)
        n_ops = sum(n.op in ("call_function", "call_method", "call_module") for n in gm.graph.nodes)
        origin = "?"
        for n in gm.graph.nodes:
            st = n.meta.get("stack_trace")
            if st:
                lines = [l.strip() for l in st.strip().splitlines() if l.strip().startswith("File")]
                user = [l for l in lines if "site-packages/torch" not in l]
                origin = short(user[0]) if user else short(lines[0])
                break
        self.graphs.append({"idx": idx, "ops": n_ops, "first_op_at": origin})

        def run(*a):
            self.launches[idx] += 1
            return gm.forward(*a)

        return run


def cmd_count(args, device):
    batch = load_batch(args.run, device, affinity=args.affinity)
    model = load_model(device, affinity=args.affinity, recycling_steps=args.recycles,
                       sampling_steps=args.steps, diffusion_samples=args.samples)
    if getattr(args, "sampler_outside", False):
        # Keep the 200-step loop in Python; Dynamo still compiles what it calls
        # (the per-step function) as its own frame.
        sm = model.structure_module
        sm.sample = torch.compiler.disable(sm.sample, recursive=False)
    torch._dynamo.reset()
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    torch._logging.set_logs(recompiles=True, graph_breaks=True)
    for name in ["torch._dynamo", "torch._dynamo.guards"]:
        logging.getLogger(name).addHandler(handler)
    backend = CountingBackend()
    compiled = torch.compile(model.forward, backend=backend)
    pa = model.predict_args
    kwargs = dict(recycling_steps=pa["recycling_steps"], num_sampling_steps=pa["sampling_steps"],
                  diffusion_samples=pa["diffusion_samples"],
                  max_parallel_samples=pa["max_parallel_samples"], run_confidence_sequentially=True)
    runs = []
    for i in range(args.repeat):
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            compiled(batch, **kwargs)
        sync(device)
        runs.append({"run": i, "seconds": time.perf_counter() - t0,
                     "graphs_compiled_so_far": len(backend.graphs),
                     "launches_this_run": sum(backend.launches.values()),
                     "launches_by_graph": dict(sorted(backend.launches.items()))})
        backend.launches.clear()
    # eager reference for the same config
    sync(device)
    t0 = time.perf_counter()
    run_forward(model, batch)
    sync(device)
    eager = time.perf_counter() - t0
    torch._logging.set_logs()
    text = log_buf.getvalue()
    recompiles = [l for l in text.splitlines() if "Recompiling function" in l or "triggered by the following guard failure" in l or "- " in l[:6]]
    return {
        "config": pa,
        "runs": runs,
        "eager_seconds": eager,
        "graphs": backend.graphs,
        "recompile_log": recompiles[:400],
    }


# ----------------------------------------------------------------------------
# modules
# ----------------------------------------------------------------------------
STRUCT_TARGETS = [
    "input_embedder", "rel_pos", "contact_conditioning", "template_module", "msa_module",
    "pairformer_module", "distogram_module", "diffusion_conditioning",
    "structure_module.score_model", "confidence_module", "bfactor_module",
]
AFF_TARGETS = ["affinity_module1"]


def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(v) for v in x)
    return x


def capture_inputs(model, batch, targets):
    """Record (args, kwargs) of the first call of each target module."""
    captured = {}
    handles = []
    for name in targets:
        mod = model.get_submodule(name)

        def hook(m, a, kw, name=name):
            if name not in captured:
                captured[name] = (_clone(a), _clone(kw))
        handles.append(mod.register_forward_pre_hook(hook, with_kwargs=True))
    # AtomDiffusion.sample is a method, not a forward: wrap it.
    sm = model.structure_module
    orig_sample = sm.sample

    def sample_wrapper(*a, **kw):
        if "structure_module.sample" not in captured:
            captured["structure_module.sample"] = (_clone(a), _clone(kw))
        return orig_sample(*a, **kw)
    sm.sample = sample_wrapper
    try:
        run_forward(model, batch)
    finally:
        sm.sample = orig_sample
        for h in handles:
            h.remove()
    return captured


def first_lines(e: BaseException, n=6) -> str:
    msg = f"{type(e).__name__}: {e}"
    lines = [l for l in msg.splitlines() if l.strip()]
    return "\n".join(lines[:n])[:1500]


def user_frame_of(e: BaseException) -> str:
    tb = traceback.extract_tb(e.__traceback__)
    user = [f for f in tb if is_user(f.filename)]
    return frame_str(user[-1]) if user else ""


def strip_nontensor_feats(a, kw):
    """torch.export only accepts pytrees of tensors/primitive leaves."""
    def fix(x):
        if isinstance(x, dict) and "token_pad_mask" in x:
            return tensor_only(x)
        return x
    return tuple(fix(v) for v in a), {k: fix(v) for k, v in kw.items()}


def cmd_modules(args, device):
    batch = load_batch(args.run, device, affinity=args.affinity)
    model = load_model(device, affinity=args.affinity, recycling_steps=0,
                       sampling_steps=2, diffusion_samples=1)
    targets = (AFF_TARGETS if args.affinity else STRUCT_TARGETS)
    if args.only:
        targets = [t for t in targets if t in args.only.split(",")]
    captured = capture_inputs(model, batch, targets + ([] if args.affinity else []))
    results = {}
    for name in targets + (["structure_module.sample"] if not args.affinity and (not args.only or "structure_module.sample" in args.only) else []):
        if name not in captured:
            results[name] = {"error": "not called"}
            continue
        a, kw = captured[name]
        if name == "structure_module.sample":
            fn = model.structure_module.sample
            mod = None
        else:
            mod = model.get_submodule(name)
            fn = mod
        entry = {}
        # 1) dynamo explain on this module alone
        torch._dynamo.reset()
        try:
            with torch.no_grad():
                ex = torch._dynamo.explain(fn)(*_clone(a), **_clone(kw))
            s = summarize_explain(ex)
            entry["dynamo"] = {k: s[k] for k in ("graph_count", "graph_break_count", "break_sites")}
        except Exception as e:  # noqa: BLE001
            entry["dynamo"] = {"error": first_lines(e), "at": user_frame_of(e)}
        # 2) torch.export (needs an nn.Module)
        if mod is not None:
            ea, ekw = strip_nontensor_feats(*_clone((a, kw)))
            for strict in (False, True):
                key = "export_strict" if strict else "export_nonstrict"
                torch._dynamo.reset()
                t0 = time.perf_counter()
                try:
                    with torch.no_grad():
                        ep = torch.export.export(mod, ea, ekw, strict=strict)
                    n_nodes = sum(1 for n in ep.graph.nodes if n.op == "call_function")
                    entry[key] = {"ok": True, "call_function_nodes": n_nodes,
                                  "seconds": round(time.perf_counter() - t0, 1)}
                except Exception as e:  # noqa: BLE001
                    entry[key] = {"ok": False, "error": first_lines(e), "at": user_frame_of(e),
                                  "seconds": round(time.perf_counter() - t0, 1)}
        results[name] = entry
        print(f"[modules] {name}: dynamo graphs={entry['dynamo'].get('graph_count')} "
              f"breaks={entry['dynamo'].get('graph_break_count')} "
              f"export_nonstrict={entry.get('export_nonstrict', {}).get('ok')} "
              f"export_strict={entry.get('export_strict', {}).get('ok')}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["explain", "count", "modules"])
    p.add_argument("--run", required=True)
    p.add_argument("--affinity", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--recycles", type=int, default=None)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--only", default=None)
    p.add_argument("--sampler-outside", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_grad_enabled(False)
    device = pick_device(args.device)
    res = {"explain": cmd_explain, "count": cmd_count, "modules": cmd_modules}[args.cmd](args, device)
    res = {"device": str(device), "torch": torch.__version__, **res} if isinstance(res, dict) else res
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
