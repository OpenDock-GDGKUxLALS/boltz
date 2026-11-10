"""Load Boltz-2 and one real featurized batch the same way `boltz predict` does.

Run `boltz predict <yaml> --out_dir runs/<name>` once first so the processed
inputs exist, then point this harness at runs/<name>/boltz_results_<stem>.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz2 import Boltz2

CACHE = Path("~/.boltz").expanduser()


def require_graph_fixes() -> None:
    """Exit unless the imported boltz has the graph-fixes changes (that branch's src/)."""
    import boltz
    from boltz.model.modules.diffusionv2 import AtomDiffusion

    if not hasattr(AtomDiffusion, "step_schedule"):
        raise SystemExit(f"{boltz.__file__} is upstream Boltz: put the graph-fixes src/ on PYTHONPATH")


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_batch(run_dir: str | Path, device: torch.device, affinity: bool = False) -> dict:
    """Featurize the first record exactly like Boltz2InferenceDataModule does."""
    run_dir = Path(run_dir)
    processed = run_dir / "processed"
    opt = lambda p: p if p.exists() else None  # noqa: E731
    manifest = Manifest.load(processed / "manifest.json")
    if affinity:
        manifest = Manifest([r for r in manifest.records if r.affinity])
    dm = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=(run_dir / "predictions") if affinity else processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=CACHE / "mols",
        num_workers=0,
        constraints_dir=opt(processed / "constraints"),
        template_dir=opt(processed / "templates"),
        extra_mols_dir=opt(processed / "mols"),
        override_method="other" if affinity else None,
        affinity=affinity,
    )
    batch = next(iter(dm.predict_dataloader()))
    return dm.transfer_batch_to_device(batch, device, 0)


def load_model(
    device: torch.device,
    affinity: bool = False,
    recycling_steps: int | None = None,
    sampling_steps: int = 200,
    diffusion_samples: int | None = None,
    use_kernels: bool = False,
) -> Boltz2:
    """Same constructor arguments as boltz.main.predict()."""
    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = 1.5
    steering = BoltzSteeringParams()
    if affinity:
        steering.fk_steering = False
        steering.physical_guidance_update = False
        steering.contact_guidance_update = False
    predict_args = {
        "recycling_steps": recycling_steps if recycling_steps is not None else (5 if affinity else 3),
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples if diffusion_samples is not None else (3 if affinity else 1),
        "max_parallel_samples": 1 if affinity else None,
        "write_confidence_summary": not affinity,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    extra = {"affinity_mw_correction": False} if affinity else {"use_kernels": use_kernels}
    model = Boltz2.load_from_checkpoint(
        CACHE / ("boltz2_aff.ckpt" if affinity else "boltz2_conf.ckpt"),
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)),
        steering_args=asdict(steering),
        **extra,
    )
    model.eval().to(device)
    # Lightning's setup("predict") turns kernels off when there is no sm80+ GPU.
    if not (torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8):
        model.use_kernels = False
    return model


def run_forward(model: Boltz2, batch: dict) -> dict:
    """The exact call predict_step makes."""
    pa = model.predict_args
    with torch.no_grad():
        return model(
            batch,
            recycling_steps=pa["recycling_steps"],
            num_sampling_steps=pa["sampling_steps"],
            diffusion_samples=pa["diffusion_samples"],
            max_parallel_samples=pa["max_parallel_samples"],
            run_confidence_sequentially=True,
        )


class StageTimer:
    """Wall-clock time per top-level submodule, synchronized on the device."""

    def __init__(self, model: torch.nn.Module, device: torch.device, names: list[str]):
        self.device = device
        self.totals = defaultdict(float)
        self.calls = defaultdict(int)
        self._start = {}
        self._handles = []
        for name in names:
            mod = model.get_submodule(name)
            self._handles.append(mod.register_forward_pre_hook(self._pre(name)))
            self._handles.append(mod.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(*_):
            sync(self.device)
            self._start[name] = time.perf_counter()
        return hook

    def _post(self, name):
        def hook(*_):
            sync(self.device)
            self.totals[name] += time.perf_counter() - self._start.pop(name)
            self.calls[name] += 1
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()


@contextmanager
def timed(label: str, device: torch.device, out: dict):
    sync(device)
    t0 = time.perf_counter()
    yield
    sync(device)
    out[label] = time.perf_counter() - t0


def describe(batch: dict) -> list[str]:
    rows = []
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            rows.append(f"{k:32s} {str(tuple(v.shape)):28s} {str(v.dtype):16s} {v.device}")
        else:
            rows.append(f"{k:32s} <{type(v).__name__}>")
    return rows
