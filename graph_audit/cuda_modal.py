"""PR #1 (graph-fixes) on a CUDA GPU with Modal: export parity, AOTInductor, cuEquivariance
kernels, and cold starts with GPU memory snapshots. Section 7 of REPORT.md.

    M=../.venv-modal/bin/modal                     # from graph_audit/
    $M run cuda_modal.py::setup                    # Boltz-2 weights into the Volume (CPU container)
    $M run cuda_modal.py::main --stage parity            # then aoti, timing, kernels, bf16 (cuda_bench.py)
    $M run cuda_modal.py::main --stage stepprobe --gpu A10G   # a stage on another GPU type
    $M run cuda_modal.py::debug --script scratch/gxx_gdb.sh   # replay a failed compile under gdb (CPU)
    $M deploy cuda_modal.py                        # snapshots need a deployed app
    ../.venv-modal/bin/python cuda_modal.py stage divprobe          # a stage on the deployed app
    ../.venv-modal/bin/python cuda_modal.py coldstart [Class ...]   # appends to reports/cuda_coldstart.json
    $M app stop boltz-export-cuda --yes && $M volume delete boltz-export --yes

Results land in reports/cuda_<stage>.json.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

FORK = "https://github.com/OpenDock-GDGKUxLALS/boltz.git"
UPSTREAM = "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc"  # jwohlwend/boltz main
FIXES = "99c12a6dde875cdad097d3bde487c87fbf725f97"  # PR #1 head
GPU = os.environ.get("BOLTZ_GPU", "L40S")
WEIGHTS_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt"
WEIGHTS_SHA256 = "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"
CACHE = "/root/.boltz"  # harness.py reads the weights from ~/.boltz
ART = f"{CACHE}/artifacts"
HERE = Path(__file__).parent
FILES = ["harness.py", "compare.py", "export_full.py", "cuda_bench.py", "reports/batch_boltz_results_prot.pt"]

app = modal.App("boltz-export-cuda")
vol = modal.Volume.from_name("boltz-export", create_if_missing=True, version=2)

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "clang", "libomp-dev")
    .pip_install("torch==2.15.0", "setuptools", "wheel")
    .run_commands(
        f"git clone {FORK} /opt/boltz && git -C /opt/boltz checkout {FIXES}",
        f"git -C /opt/boltz worktree add /opt/boltz-upstream {UPSTREAM}",
        "pip install --no-build-isolation fairscale==0.4.13",
        "pip install /opt/boltz",  # graph-fixes is the installed boltz; upstream runs via PYTHONPATH
    )
    .pip_install("cuequivariance_torch==0.13.0", "cuequivariance_ops_torch_cu13==0.13.0")
    # clang++ for the AOTInductor wrapper: g++ 13 overflows Modal's 8 MiB stack on it (REPORT section 7)
    # x86-64-v3 (AVX2), not -march=native: a package built on an AVX-512 host dies with SIGILL on others
    .env({"CXX": "clang++", "TORCHINDUCTOR_CPP_MARCH": "x86-64-v3", "PYTHONUNBUFFERED": "1", "ARTIFACTS": ART, "UPSTREAM_SRC": "/opt/boltz-upstream/src",
          "BOLTZ_FIXES_COMMIT": FIXES})
)
debug_image = image.apt_install("gdb", "time")  # no local files: it works on the Volume
for f in FILES:
    image = image.add_local_file(HERE / f, f"/root/ga/{f}")
GPU_OPTS = dict(image=image, gpu=GPU, volumes={CACHE: vol})


@app.function(image=image, volumes={CACHE: vol}, cpu=2, memory=4096, timeout=1800)
def setup() -> str:
    import hashlib
    import urllib.request

    def sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        return h.hexdigest()

    dst = Path(CACHE) / "boltz2_conf.ckpt"
    if not dst.exists() or sha256(dst) != WEIGHTS_SHA256:
        urllib.request.urlretrieve(WEIGHTS_URL, f"{dst}.part")  # noqa: S310
        os.replace(f"{dst}.part", dst)
    assert sha256(dst) == WEIGHTS_SHA256
    Path(ART).mkdir(exist_ok=True)
    vol.commit()
    return f"{dst}: sha256 ok"


@app.function(cpu=8, memory=49152, timeout=4 * 3600, **GPU_OPTS)
def bench(stage: str) -> dict:
    result = Path(ART) / f"{stage}.json"
    result.unlink(missing_ok=True)
    proc = subprocess.run([sys.executable, "cuda_bench.py", stage], cwd="/root/ga")
    vol.commit()
    return {"returncode": proc.returncode, **(json.loads(result.read_text()) if result.exists() else {})}


@app.function(image=debug_image, volumes={CACHE: vol}, cpu=8, memory=65536, timeout=2 * 3600)
def shell(script: str) -> str:
    """Run a bash script next to the Volume (CPU only), e.g. to replay a crashing compile under gdb."""
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    vol.commit()
    return f"exit {proc.returncode}\n{proc.stdout[-40000:]}\n--- stderr ---\n{proc.stderr[-20000:]}"


@app.local_entrypoint()
def debug(script: str, gpu: str = ""):
    print((shell.with_options(gpu=gpu) if gpu else shell).remote(Path(script).read_text()))


@app.local_entrypoint()
def main(stage: str = "parity", gpu: str = ""):
    out = HERE / "reports" / f"cuda_{stage}.json"
    out.write_text(json.dumps((bench.with_options(gpu=gpu) if gpu else bench).remote(stage), indent=1))
    print("wrote", out)


# Cold-start test. Every call gets a new container (single_use_containers).
COLD_OPTS = dict(cpu=4, memory=32768, single_use_containers=True, timeout=1800, startup_timeout=1800, **GPU_OPTS)
SNAPSHOT = dict(enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True})


class Worker:
    kind = ""

    def start(self):
        sys.path.insert(0, "/root/ga")
        os.chdir("/root/ga")
        t0 = time.perf_counter()
        import torch  # noqa: F401
        from cuda_bench import Engine

        self.engine = Engine(self.kind)
        self.engine.t["import_s"] = round(time.perf_counter() - t0, 2)
        self.engine.boot()

    @modal.method()
    def predict(self) -> dict:
        return self.engine.predict()


class SnapshotWorker(Worker):
    @modal.enter(snap=True)
    def boot(self):
        self.start()

    @modal.enter(snap=False)
    def restored(self):
        self.engine.ping()  # the first GPU op after a restore; fails here if the CUDA state is broken


class BaselineWorker(Worker):
    @modal.enter()
    def boot(self):
        self.start()


@app.cls(**SNAPSHOT, **COLD_OPTS)
class AOTISnapshot(SnapshotWorker):
    kind = "aoti"


@app.cls(**SNAPSHOT, **COLD_OPTS)
class EagerSnapshot(SnapshotWorker):
    kind = "eager"


@app.cls(**COLD_OPTS)
class AOTIBaseline(BaselineWorker):
    kind = "aoti"


@app.cls(**COLD_OPTS)
class EagerBaseline(BaselineWorker):
    kind = "eager"


def coldstart(names, trials=2):
    """Call each deployed class `trials` times (+2 for snapshot classes: the first starts take the snapshots)."""
    out = HERE / "reports" / "cuda_coldstart.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    for name in names:
        cls = modal.Cls.from_name(app.name, name)
        for _ in range(trials + 2 * name.endswith("Snapshot")):
            t0 = time.time()
            r = cls().predict.remote()
            r.update(cls=name, client_s=round(time.time() - t0, 2))
            r["cold_penalty_s"] = round(r["client_s"] - r["predict_s"], 2)
            print(name, r["kind"], r["client_s"], r["predict_s"], r["same_as_boot"], r["boot"], flush=True)
            rows.append(r)
            out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__" and sys.argv[1:2] == ["coldstart"]:
    coldstart(sys.argv[2:] or ["EagerBaseline", "AOTIBaseline", "EagerSnapshot", "AOTISnapshot"])
elif __name__ == "__main__" and sys.argv[1:2] == ["stage"]:  # a bench stage on the deployed app
    result = modal.Function.from_name(app.name, "bench").remote(sys.argv[2])
    (HERE / "reports" / f"cuda_{sys.argv[2]}.json").write_text(json.dumps(result, indent=1))
