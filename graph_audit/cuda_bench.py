"""Boltz-2 export checks on a CUDA GPU. Runs inside the Modal container (see cuda_modal.py).

    python cuda_bench.py parity    # upstream vs graph-fixes eager (compare.py), then export_full.py --strict
    python cuda_bench.py aoti      # strict export -> AOTInductor package; compiled vs eager
    python cuda_bench.py aoti_eager_rng   # the same, compiled with eager's random ops (numeric check)
    python cuda_bench.py timing    # warm runs: eager fp32 / bf16-mixed / + cuEquivariance, AOTInductor
    python cuda_bench.py kernels   # strict export with the cuEquivariance kernels (use_kernels=True)
    python cuda_bench.py bf16      # strict export traced under bf16 autocast (small config)
    python cuda_bench.py divprobe  # why graph-fixes eager is not bit-identical to upstream on CUDA
    python cuda_bench.py stepprobe # first sampling step where upstream and graph-fixes diverge
    python cuda_bench.py bf16debug # pdb post-mortem of the bf16 replay failure; autocast inside the program
    python cuda_bench.py steptrace # pdb breakpoints on the sampler update: operands and result, both versions

Each stage writes $ARTIFACTS/<stage>.json. Input: the 136-token protein batch of section 3;
3 recycles, 200 steps, 2 samples unless noted. `Engine` is the worker of the cold-start test.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

ART = Path(os.environ.get("ARTIFACTS", "/root/.boltz/artifacts"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ART.parent / "inductor-cache"))  # kept across containers

import torch  # noqa: E402

BATCH = "reports/batch_boltz_results_prot.pt"
AOTI = ART / "prot_aoti.pt2"
RUN = "runs/cli_prot/boltz_results_prot"  # only names the cached batch for compare.py


def env() -> dict:
    import boltz
    import triton

    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "capability": f"{p.major}.{p.minor}", "torch": torch.__version__,
            "cuda": torch.version.cuda, "triton": triton.__version__, "boltz": boltz.__file__,
            "fixes_commit": os.environ.get("BOLTZ_FIXES_COMMIT")}


def feats() -> dict:
    return {k: v.cuda() for k, v in torch.load(BATCH).items()}


def program(recycles=3, steps=200, samples=2, kernels=False):
    from export_full import Boltz2Program
    from harness import load_model

    model = load_model(torch.device("cuda"), recycling_steps=recycles, sampling_steps=steps,
                       diffusion_samples=samples, use_kernels=kernels)
    model.requires_grad_(False)
    return Boltz2Program(model).eval()


def run(fn, x, seed=0, bf16=False):
    """One synchronized call; bf16=True is Lightning's bf16-mixed (autocast) as in `boltz predict`."""
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), ctx:
        out = fn(x)
    torch.cuda.synchronize()
    return out, round(time.perf_counter() - t0, 3)


def diff(a, b) -> dict:
    out = {k: (a[k].float() - b[k].float()).abs().max().item() for k in a if k in b}
    out["coords_rmsd"] = ((a["sample_atom_coords"] - b["sample_atom_coords"]) ** 2).sum(-1).mean().sqrt().item()
    return out


def scores(out) -> dict:
    return {k: [round(v, 4) for v in out[k].flatten().tolist()]
            for k in ("ptm", "iptm", "complex_plddt", "confidence_score")}


def digest(out) -> str:
    h = hashlib.sha256()
    for k in sorted(out):
        h.update(out[k].detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def error(e: BaseException) -> str:
    """First line of the exception plus any compiler error lines (with the two lines after each)."""
    lines = "".join(traceback.format_exception_only(e)).strip().splitlines()
    hits = [i for i, line in enumerate(lines) if i and "error" in line.lower()][:6]
    keep = sorted({0, *(j for i in hits for j in range(i, min(i + 3, len(lines))))})
    return "\n".join(lines[i][:400] for i in keep)


def graphs(ep):
    return [m for m in ep.graph_module.modules() if isinstance(m, torch.fx.GraphModule)]


def export(prog, x):
    t0 = time.perf_counter()
    ep = torch.export.export(prog, (x,), strict=True)
    return ep, round(time.perf_counter() - t0, 1)


def aoti_compile(ep, path):
    """The wrapper is built with clang++ (CXX in the image) at -O0: it only launches kernels, and g++ 13
    overflows the sandbox's 8 MiB stack on it."""
    import torch._inductor.config as inductor_config

    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"
    t0 = time.perf_counter()
    path = torch._inductor.aoti_compile_and_package(ep, package_path=str(path))
    return path, round(time.perf_counter() - t0, 1)


def write(stage: str, report: dict) -> None:
    text = json.dumps(report, indent=1, default=str)
    (ART / f"{stage}.json").write_text(text)
    print(text, flush=True)


def parity():
    report = {"env": env()}
    saves = {}
    for name, path in (("upstream", os.environ["UPSTREAM_SRC"]), ("upstream_again", os.environ["UPSTREAM_SRC"]),
                       ("graph_fixes", "")):
        out = f"reports/cuda_out_{name}.pt"
        subprocess.run([sys.executable, "compare.py", "save", "--run", RUN, "--samples", "2", "--out", out],
                       check=True, env={**os.environ, "PYTHONPATH": path})
        saves[name] = torch.load(out)
    up, again, fixed = saves["upstream"], saves["upstream_again"], saves["graph_fixes"]
    keys = [k for k in up if k != "_meta" and k in fixed]
    report["src"] = {n: s["_meta"]["boltz_src"] for n, s in saves.items()}
    report["eager_seconds"] = {n: round(s["_meta"]["seconds"], 2) for n, s in saves.items()}
    report["upstream_vs_upstream"] = {k: (up[k] - again[k]).abs().max().item() for k in keys}
    report["upstream_vs_fixes"] = {k: (up[k] - fixed[k]).abs().max().item() for k in keys}
    out = str(ART / "export_full_cuda.json")
    subprocess.run([sys.executable, "export_full.py", "--strict", "--batch", BATCH, "--samples", "2",
                    "--out", out], check=True)
    report["export_full"] = json.loads(Path(out).read_text())
    write("parity", report)


def aoti(eager_rng=False):
    """eager_rng: the compiled code calls eager's random ops, so its samples can be compared with
    eager. Inductor keeps their order with control_deps, which its scan lowering can't re-trace in
    torch 2.15, so that pass is switched off here and the order is left to the scheduler."""
    import torch._inductor.config as inductor_config
    from torch._inductor.fx_passes import post_grad

    name = "aoti_eager_rng" if eager_rng else "aoti"
    if eager_rng:
        inductor_config.fallback_random = True
        post_grad._chain_random_ops_for_ordering = lambda graph: None  # noqa: SLF001
    report = {"env": env(), "eager_rng": eager_rng}
    x = feats()
    prog = program()
    ref, report["eager_s"] = run(prog, x)
    ep, report["export_s"] = export(prog, x)
    if not eager_rng:
        try:
            torch.export.save(ep, str(ART / "prot_ep.pt2"))
            report["export_save"] = "ok"
        except Exception as e:  # noqa: BLE001
            report["export_save"] = error(e)
    path, report["compile_s"] = aoti_compile(ep, ART / f"prot_{name}.pt2")
    report["package_gb"] = round(Path(path).stat().st_size / 1e9, 2)
    del ep
    t0 = time.perf_counter()
    runner = torch._inductor.aoti_load_package(path)
    report["load_s"] = round(time.perf_counter() - t0, 1)
    first, report["first_run_s"] = run(runner, x)
    got, report["second_run_s"] = run(runner, x)
    report["vs_eager"] = diff(got, ref)
    report["repeat"] = diff(first, got)
    report["scores"] = {"eager": scores(ref), "aoti": scores(got)}
    write(name, report)


def timing(n=3):
    report = {"env": env(), "runs": n, "config": "136 tokens, 3 recycles, 200 steps, 2 samples"}
    x = feats()
    prog = program()
    runner = torch._inductor.aoti_load_package(str(AOTI))
    cases = [  # name, callable, cuEquivariance kernels, bf16-mixed, fp32 matmul precision
        ("eager fp32", prog, False, False, "highest"),
        ("eager fp32, TF32 matmuls", prog, False, False, "high"),
        ("eager bf16-mixed", prog, False, True, "highest"),
        ("eager bf16-mixed + cuEquivariance (boltz predict)", prog, True, True, "highest"),
        ("AOTInductor fp32", runner, False, False, "highest"),
        ("AOTInductor fp32, TF32 matmuls", runner, False, False, "high"),
    ]
    for name, fn, kernels, bf16, precision in cases:
        prog.model.use_kernels = kernels
        torch.set_float32_matmul_precision(precision)
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        try:
            times = []
            for _ in range(n + 1):  # the first call is a warm-up
                out, t = run(fn, x, bf16=bf16)
                times.append(t)
            report[name] = {"first_s": times[0], "median_s": statistics.median(times[1:]), "runs_s": times[1:],
                            "work_gb": round((torch.cuda.max_memory_allocated() - base) / 1e9, 2),
                            "scores": scores(out)}
        except Exception as e:  # noqa: BLE001
            report[name] = {"error": error(e)}
        print(name, report[name], flush=True)
    torch.set_float32_matmul_precision("highest")
    write("timing", report)


def kernels():
    """Small config (1 recycle, 3 steps, 1 sample): only whether it traces and lowers."""
    import boltz.model.layers.triangular_attention.primitives as tri_attn
    import boltz.model.layers.triangular_mult as tri_mult

    report = {"env": env(), "config": "1 recycle, 3 steps, 1 sample"}
    x = feats()
    prog = program(recycles=1, steps=3, samples=1, kernels=True)
    report["use_kernels"] = prog.model.use_kernels
    for bf16 in (False, True):
        try:
            report[f"eager_{'bf16' if bf16 else 'fp32'}_s"] = run(prog, x, bf16=bf16)[1]
        except Exception as e:  # noqa: BLE001
            report[f"eager_{'bf16' if bf16 else 'fp32'}_s"] = error(e)
    try:
        export(prog, x)
        report["export_as_is"] = "ok"
    except Exception as e:  # noqa: BLE001
        report["export_as_is"] = error(e)
    # The kernels are wrapped in torch.compiler.disable (E1); try them without it.
    tri_mult.kernel_triangular_mult = inspect.unwrap(tri_mult.kernel_triangular_mult)
    tri_attn.kernel_triangular_attn = inspect.unwrap(tri_attn.kernel_triangular_attn)
    try:
        ep, report["export_unwrapped_s"] = export(prog, x)
        report["kernel_ops"] = sorted({str(n.target) for g in graphs(ep) for n in g.graph.nodes
                                       if n.op == "call_function" and "cuequivariance" in str(n.target)})
        report["kernel_registrations"] = {  # C++ (TORCH_LIBRARY) or Python (torch.library) kernels
            op: [line.strip()[:200] for line in torch._C._dispatch_dump("::".join(op.split(".")[:2])).splitlines()
                 if line.startswith(("CUDA", "Meta", "Python"))] for op in report["kernel_ops"]}
        ref = run(prog, x)[0]
        report["exported_vs_eager"] = diff(run(ep.module(), x)[0], ref)
        try:
            path, report["compile_s"] = aoti_compile(ep, ART / "small_kernels_aoti.pt2")
            report["aoti_vs_eager"] = diff(run(torch._inductor.aoti_load_package(path), x)[0], ref)
        except Exception as e:  # noqa: BLE001
            report["aoti"] = error(e)
    except Exception as e:  # noqa: BLE001
        report["export_unwrapped"] = error(e)
    write("kernels", report)


def bf16():
    """Section 5 item 4 on CUDA: a strict export traced under bf16 autocast (2 passes, 2 steps)."""
    report = {"env": env(), "config": "1 recycle, 2 steps, 2 samples, bf16 autocast"}
    x = feats()
    prog = program(recycles=1, steps=2, samples=2)
    ref = run(prog, x, bf16=True)[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ep, report["export_s"] = export(prog, x)
    report["scans"] = sum(1 for n in ep.graph.nodes if n.op == "call_function" and "scan" in str(n.target))
    report["exported_vs_eager"] = diff(run(ep.module(), x, bf16=True)[0], ref)
    write("bf16", report)


def divprobe():
    """CUDA divides by a host scalar s as x * (1/s). Upstream divides by t_hat as a Python float;
    graph-fixes divides by a device tensor (true division). Rerun graph-fixes with x * (1 / t_hat)."""
    import argparse
    import textwrap

    import compare
    from boltz.model.modules import diffusionv2

    report = {"env": env()}
    outs = {}
    for name, path in (("upstream", os.environ["UPSTREAM_SRC"]), ("graph_fixes", "")):
        outs[name] = f"reports/cuda_out_{name}.pt"
        subprocess.run([sys.executable, "compare.py", "save", "--run", RUN, "--samples", "2", "--out", outs[name]],
                       check=True, env={**os.environ, "PYTHONPATH": path})
    src = textwrap.dedent(inspect.getsource(diffusionv2.AtomDiffusion.denoise_step))
    old = "(noisy - denoised) / t_hat"
    assert src.count(old) == 1
    scope = {}
    exec(src.replace(old, "(noisy - denoised) * (1 / t_hat)"), vars(diffusionv2), scope)  # noqa: S102
    diffusionv2.AtomDiffusion.denoise_step = scope["denoise_step"]
    outs["graph_fixes_reciprocal"] = "reports/cuda_out_graph_fixes_reciprocal.pt"
    compare.save(argparse.Namespace(run=RUN, affinity=False, recycles=None, steps=200, samples=2, seed=0,
                                    out=outs["graph_fixes_reciprocal"]))
    up = torch.load(outs["upstream"])
    batch = torch.load(BATCH)
    real = batch["atom_pad_mask"][0].bool()
    rep = batch["token_to_rep_atom"][0].sum(0).bool()  # the atoms confidence reads distances from
    for name in ("graph_fixes", "graph_fixes_reciprocal"):
        other = torch.load(outs[name])
        report[f"upstream_vs_{name}"] = {k: (up[k] - other[k]).abs().max().item()
                                         for k in up if k != "_meta" and k in other}
        d = (up["sample_atom_coords"] - other["sample_atom_coords"]).abs().amax(dim=(0, 2))
        report[f"coords_{name}"] = {"real_atoms": d[real].max().item(), "padding_atoms": d[~real].max().item(),
                                    "representative_atoms": d[rep].max().item(), "atoms_that_differ": int((d > 0).sum()),
                                    "real": int(real.sum()), "padding": int((~real).sum())}
    write("divprobe", report)


def record(out):
    """Run compare.py's prediction (5 steps) saving each score-network call: input, noise level, output."""
    import argparse

    import compare
    from boltz.model.modules import diffusionv2

    forward, calls = diffusionv2.AtomDiffusion.preconditioned_network_forward, []

    def recorded(self, x, sigma, network_condition_kwargs):
        y = forward(self, x, sigma, network_condition_kwargs)
        calls.append([t.detach().float().cpu() for t in (x, torch.as_tensor(sigma).reshape(-1), y)])
        return y

    diffusionv2.AtomDiffusion.preconditioned_network_forward = recorded
    compare.save(argparse.Namespace(run=RUN, affinity=False, recycles=None, steps=5, samples=2, seed=0, out=out))
    torch.save(calls, out + ".calls")


def stepprobe():
    report = {"env": env()}
    calls = {}
    for name, path in (("upstream", os.environ["UPSTREAM_SRC"]), ("graph_fixes", "")):
        out = f"reports/cuda_steps_{name}.pt"
        subprocess.run([sys.executable, "cuda_bench.py", "_record", out], check=True,
                       env={**os.environ, "PYTHONPATH": path})
        calls[name] = torch.load(out + ".calls")
    report["steps"] = [{"input": (a[0] - b[0]).abs().max().item(), "sigma": (a[1] - b[1]).abs().max().item(),
                        "output": (a[2] - b[2]).abs().max().item(), "sigma_value": a[1].tolist()}
                       for a, b in zip(calls["upstream"], calls["graph_fixes"])]
    write("stepprobe", report)


PDB_BF16 = """up
list
!_t = {k: (v.dtype, tuple(v.shape)) for k, v in dict(locals()).items() if isinstance(v, torch.Tensor)}
p _t
p [(n.name, str(n.args[0]), n.kwargs.get("dtype"), (n.args[0].meta.get("stack_trace") or "")[-900:]) for n in self.graph.nodes if "assert_tensor_metadata" in str(n.target)]
up 9
p [str(n.args[:4]) for n in self.graph.nodes if "wrap_with_autocast" in str(n.target)][:3]
quit
"""


def postmortem(e: BaseException) -> str:
    """Scripted pdb on the traceback: the failing graph line, its tensors, and the Boltz source of its asserts."""
    import io
    import pdb

    out = io.StringIO()
    debugger = pdb.Pdb(stdin=io.StringIO(PDB_BF16), stdout=out)
    debugger.use_rawinput = False
    debugger.reset()
    debugger.interaction(None, e.__traceback__)
    text = out.getvalue()
    return text if len(text) < 14000 else text[:10000] + "\n[...]\n" + text[-4000:]


class Autocast(torch.nn.Module):
    """The program with Lightning's bf16-mixed inside it, so the export records its own precision."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, feats):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.inner(feats)


def bf16debug():
    report = {"env": env(), "config": "0 recycles (1 trunk pass), 2 steps, 1 sample"}
    x = feats()
    prog = program(recycles=0, steps=2, samples=1)
    ref = run(prog, x, bf16=True)[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ep, report["export_outer_autocast_s"] = export(prog, x)
    for name, bf16 in (("replay_under_autocast", True), ("replay_without_autocast", False)):
        try:
            report[name] = diff(run(ep.module(), x, bf16=bf16)[0], ref)
        except Exception as e:  # noqa: BLE001
            report[name], report[name + "_pdb"] = error(e), postmortem(e)
    wrapped = Autocast(prog)
    ep, report["export_inner_autocast_s"] = export(wrapped, x)
    report["inner_autocast_hops"] = sum(1 for g in graphs(ep) for n in g.graph.nodes if "autocast" in str(n.target))
    report["scans"] = sum(1 for g in graphs(ep) for n in g.graph.nodes if "scan" in str(n.target))
    try:
        report["inner_autocast_vs_eager"] = diff(run(ep.module(), x)[0], run(wrapped, x)[0])
    except Exception as e:  # noqa: BLE001
        report["inner_autocast_vs_eager"], report["inner_autocast_pdb"] = error(e), postmortem(e)
    write("bf16debug", report)


UPDATE = {  # the sampler's update line in each version, and its operands to save there
    "denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat":
        "dict(noisy=atom_coords_noisy, denoised=atom_coords_denoised, t_hat=torch.tensor(t_hat, dtype=torch.float64),"
        " coef=torch.tensor(step_scale * (sigma_t - t_hat), dtype=torch.float64))",
    "return noisy + step_coef * ((noisy - denoised) / t_hat)": "dict(noisy=noisy, denoised=denoised, t_hat=t_hat, coef=step_coef)",
}
FORMULAS = {  # each version's update from saved operands; t64, c64 are upstream's float64 values
    "upstream": lambda n, d, t, c, t64, c64: n + c64 * ((n - d) / t64),
    "graph_fixes": lambda n, d, t, c, t64, c64: n + c * ((n - d) / t),
    "graph_fixes_reciprocal": lambda n, d, t, c, t64, c64: n + c * ((n - d) * (1 / t)),
    "graph_fixes_reciprocal64": lambda n, d, t, c, t64, c64: n + c * ((n - d) * torch.tensor(1 / t64, device=n.device)),
}
PATCHES = {  # TRACE_VARIANT: (function, old, new) rewrites of graph-fixes
    "reciprocal": [("denoise_step", "((noisy - denoised) / t_hat)", "((noisy - denoised) * (1 / t_hat))")],
    "reciprocal64": [  # 1 / t_hat in float64 on the host, as CUDA does for upstream's Python float
        ("step_schedule", "return t_hat, noise_std, step_scale * (sigma_t - t_hat)",
         "return t_hat, noise_std, step_scale * (sigma_t - t_hat), 1 / t_hat"),
        ("denoise_step", "t_hat, noise_std, step_coef = scalars", "t_hat, noise_std, step_coef, inv_t_hat = scalars"),
        ("denoise_step", "((noisy - denoised) / t_hat)", "((noisy - denoised) * inv_t_hat)"),
    ],
}


def patch(variant):
    """Rewrite graph-fixes methods, compiled against the file's own line numbers so breakpoints still apply."""
    import textwrap

    from boltz.model.modules import diffusionv2

    sources = {}
    for name, old, new in PATCHES[variant]:
        fn = getattr(diffusionv2.AtomDiffusion, name)
        src = sources.get(name) or ("\n" * (fn.__code__.co_firstlineno - 1) + textwrap.dedent(inspect.getsource(fn)))
        assert src.count(old) == 1, (name, old)
        sources[name] = src.replace(old, new)
    for name, src in sources.items():
        scope = {}
        exec(compile(src, diffusionv2.__file__, "exec"), vars(diffusionv2), scope)  # noqa: S102
        setattr(diffusionv2.AtomDiffusion, name, scope[name])


def trace(out):
    """compare.py's prediction (200 steps) under pdb, with a breakpoint on the update line that saves its
    operands. TRACE_VARIANT rewrites graph-fixes' update first (PATCHES)."""
    import argparse
    import builtins
    import io
    import pdb

    import compare
    from boltz.model.modules import diffusionv2

    lines = Path(diffusionv2.__file__).read_text().splitlines()
    line, save = next((i, v) for i, text in enumerate(lines, 1) for k, v in UPDATE.items() if k in text)
    if os.environ.get("TRACE_VARIANT"):
        patch(os.environ["TRACE_VARIANT"])
    builtins._dump = []
    # No "!" before the statement: pdb drops "!" lines from a commands list.
    cmds = f"""b {diffusionv2.__file__}:{line}
commands
silent
_dump.append({{k: v.detach().cpu() for k, v in {save}.items()}})
c
c
"""
    log = io.StringIO()
    debugger = pdb.Pdb(stdin=io.StringIO(cmds), stdout=log)
    debugger.use_rawinput = False
    debugger.runcall(compare.save, argparse.Namespace(run=RUN, affinity=False, recycles=None, steps=200, samples=2,
                                                      seed=0, out=out))
    torch.save({"steps": builtins._dump, "pdb": log.getvalue()[-3000:], "break": f"{diffusionv2.__file__}:{line}"},
               out + ".trace")


def steptrace():
    """200 steps: the first step where each version's update departs from upstream's, and the final outputs."""
    report = {"env": env()}
    dumps, outs = {}, {}
    versions = {"upstream": (os.environ["UPSTREAM_SRC"], ""), "graph_fixes": ("", ""),
                "graph_fixes_reciprocal": ("", "reciprocal"), "graph_fixes_reciprocal64": ("", "reciprocal64")}
    for name, (path, variant) in versions.items():
        out = f"reports/cuda_trace_{name}.pt"
        subprocess.run([sys.executable, "cuda_bench.py", "_trace", out], check=True,
                       env={**os.environ, "PYTHONPATH": path, "TRACE_VARIANT": variant})
        dumps[name], outs[name] = torch.load(out + ".trace"), torch.load(out)
        report[f"pdb_{name}"] = {"break": dumps[name]["break"], "hits": len(dumps[name]["steps"])}
    up = dumps["upstream"]["steps"]
    for name in list(versions)[1:]:
        rows = []
        for i, (a, b) in enumerate(zip(up, dumps[name]["steps"])):
            n, d, t, c = (x.cuda() for x in (b["noisy"], b["denoised"], b["t_hat"], b["coef"]))
            t64, c64 = a["t_hat"].item(), a["coef"].item()
            ref = FORMULAS["upstream"](a["noisy"].cuda(), a["denoised"].cuda(), None, None, t64, c64)
            rows.append({"step": i, "t_hat": t64, "noisy": (a["noisy"] - b["noisy"]).abs().max().item(),
                         "denoised": (a["denoised"] - b["denoised"]).abs().max().item(),
                         "update": (FORMULAS[name](n, d, t.float(), c.float(), t64, c64) - ref).abs().max().item()})
        first = next((r for r in rows if r["noisy"] or r["denoised"] or r["update"]), None)
        report[name] = {"first_difference": first,
                        "steps_with_differences": sum(1 for r in rows if r["noisy"] or r["denoised"] or r["update"]),
                        "around_first": rows[max(0, first["step"] - 1):first["step"] + 2] if first else [],
                        "final_outputs_vs_upstream": {k: (outs["upstream"][k] - outs[name][k]).abs().max().item()
                                                      for k in outs["upstream"] if k != "_meta" and k in outs[name]}}
    write("steptrace", report)


def identity() -> dict:
    return {"task_id": os.environ.get("MODAL_TASK_ID"), "time": time.time(),
            "snap_restore": os.environ.get("MODAL_ENABLE_SNAP_RESTORE") == "1"}


class Engine:
    """Worker state for the cold-start test: the AOTInductor package or the eager model, on the GPU."""

    def __init__(self, kind: str):
        self.kind, self.t = kind, {}

    def boot(self):
        t0 = time.perf_counter()
        torch.zeros(1, device="cuda")
        self.t["cuda_init_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        if self.kind == "aoti":
            self.fn, self.bf16 = torch._inductor.aoti_load_package(str(AOTI)), False
        else:  # what `boltz predict` runs on an sm80+ GPU
            self.fn, self.bf16 = program(kernels=True), True
        self.t["load_s"] = round(time.perf_counter() - t0, 2)
        self.x = feats()
        out, self.t["warmup_s"] = run(self.fn, self.x, bf16=self.bf16)
        self.ref, self.boot_id = digest(out), identity()

    def ping(self):
        t0 = time.perf_counter()
        (torch.ones(1, device="cuda") + 1).item()
        self.t["first_op_s"] = round(time.perf_counter() - t0, 3)
        self.ready_id = identity()

    def predict(self, seed=0) -> dict:
        out, seconds = run(self.fn, self.x, seed=seed, bf16=self.bf16)
        ready = getattr(self, "ready_id", self.boot_id)
        kind = ("no-snapshot" if not ready["snap_restore"] else
                "restored" if ready["task_id"] != self.boot_id["task_id"] else "snapshot-created")
        return {"kind": kind, "predict_s": seconds, "same_as_boot": digest(out) == self.ref,
                "boot": self.t, "boot_id": self.boot_id, "ready_id": ready, "scores": scores(out)}


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    ART.mkdir(parents=True, exist_ok=True)
    stages = {"parity": parity, "aoti": aoti, "aoti_eager_rng": lambda: aoti(eager_rng=True), "timing": timing,
              "kernels": kernels, "bf16": bf16, "divprobe": divprobe, "stepprobe": stepprobe,
              "bf16debug": bf16debug, "steptrace": steptrace}
    hidden = {"_record": record, "_trace": trace}
    hidden[sys.argv[1]](sys.argv[2]) if sys.argv[1] in hidden else stages[sys.argv[1]]()
