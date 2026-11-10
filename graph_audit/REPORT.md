# Boltz-2 inference: runtime map and where the torch graph breaks

Boltz 2.2.1 (`b1ebfc4`), torch 2.15.0, Python 3.12, Apple M4 Max (MPS, fp32). 2026-11-10.

## Summary

- **Put together (section 4).** The whole structure prediction now exports as **one**
  `torch.export` program: embedding, 4 recycles, conditioning, 200 sampling steps,
  confidence including pTM/ipTM/chain-pair ipTM, and the ranking score. Export takes 93 s
  (strict mode); the graph holds one trunk pass and one sampling step as loop bodies. Running
  it reproduces the eager model exactly. The structure → affinity hand-off no longer
  goes through a file and the CPU featurizer: a tensor version of the crop picks the same
  211 tokens and builds the same 34 input features. **Structure model + crop + affinity
  model export together as one program** (298 s), and all 21 outputs match eager exactly.
- How: confidence post-processing rewritten as tensor code; the recycling and sampling
  loops become `scan`s when traced (eager still runs a plain Python loop); the noise
  schedule is cached before tracing. Upstream vs these changes, eager: bit-identical on CUDA
  (section 7; chain-pair ipTM ≤ 2.4e-7, a different summation order). On CPU/MPS the coordinates differ by
  ≤ 6.7e-4 Å and every score is identical: upstream rounds one division per sampling step differently per device.
- **On a CUDA GPU (section 7, Modal L40S).** The strict export works and replays exactly like eager.
  AOTInductor lowers the program to Triton kernels, but g++ overflows its 8 MiB stack building the 149k-line
  wrapper (traced with gdb). The cuEquivariance kernels export once their `torch.compiler.disable` wrappers go.
  bf16-autocast exports don't replay (a torch 2.15 dtype-tracking bug, found with pdb). A GPU memory snapshot cuts
  the eager worker's cold start from about 58 s to 7.5–14 s. pdb traced a one-rounding-per-step difference from
  upstream on CUDA to how PyTorch divides by a Python float there; `99c12a6` follows it, and is bit-identical.
- **AOTInductor on the L40S:** 3.19 s per prediction in fp32 and 1.78 s with TF32, against 4.95 s for `boltz predict`'s
  eager bf16 + cuEquivariance. It needs clang++ and `-march=x86-64-v3`: both problems were found with gdb.
- Not done: AOTInductor with snapshots and bf16, shape buckets, and steering with constraints (such inputs are
  detected and should run eagerly). See section 5.

Earlier findings:

- The expensive parts of the network are already graph-friendly: the trunk (templates,
  MSA module, 64-layer Pairformer × 4 recycles) and the diffusion score model each trace
  as one graph with no breaks inside, and the Pairformer exports in strict mode. What
  breaks the graph are the joins between stages and the code around the network inside
  the diffusion loop.
- Upstream, one prediction (136-token protein, 3 recycles, 200 steps) compiles into
  **21 graphs and makes 3,754 graph launches and 1,006 GPU→CPU syncs**. Three `.item()`
  calls in each diffusion step make Dynamo skip the whole sampler, so all 200 steps run
  as Python. The conditioning module returns a `functools.partial` that the score model
  takes as input every step, so neither of them can be exported.
- Seven small fixes in six files, on branch `graph-fixes`, give **bit-identical outputs** and bring that to
  **11 graphs, 210 launches (1 trunk + 200 steps + 9 small), 6 syncs, 0 recompiles**. With them,
  the trunk, the conditioning module and one denoising step export with
  `torch.export`, and running those exported programs end to end reproduces the eager
  model (see "Exported pipeline" below).
- At that point the remaining obstacles to one exported artifact were: confidence post-processing
  (pTM/ipTM per chain, which also **silently exported as zeros**), the unrolled loops, the
  file-based affinity hand-off, steering potentials when constraints are given, the
  cuEquivariance kernels on CUDA, and input shapes. Section 4 removes the first three.

## 1. How one prediction runs

`boltz predict` parses the YAML, builds MSAs (optional server), featurizes on CPU in the
DataLoader (`Boltz2InferenceDataModule`, batch size 1, no padding to fixed sizes), then
Lightning calls `Boltz2.predict_step → Boltz2.forward` (`src/boltz/model/models/boltz2.py:402`):

| # | Stage | Code | Runs |
|---|---|---|---|
| 1 | Input embedder: atom encoder + atom attention → token features `s_inputs` | `trunkv2.py:173` | 1× |
| 2 | Pair init: outer sum + relative position + bonds + contact conditioning | `boltz2.py:421-430` | 1× |
| 3 | Recycling: template module (2 layers), MSA module (4 layers), Pairformer (64 layers) | `boltz2.py:440-491` | 4× (`recycling_steps=3`) |
| 4 | Diffusion conditioning: atom encoder with trunk features, attention biases for 3+3 atom layers and 24 token layers | `diffusion_conditioning.py:83` | 1× |
| 5 | `AtomDiffusion.sample`: per step random rotation, noise, score model (3 atom enc + 24 token transformer + 3 atom dec layers), rigid alignment (SVD), Euler update; plus contact/template steering potentials | `diffusionv2.py:295` | 200 steps |
| 6 | Confidence module: 8-layer Pairformer on trunk `z` + predicted-structure distogram, heads, pTM/ipTM per chain pair | `confidencev2.py:109` | once per sample (Python loop) |
| 7 | Affinity (second checkpoint, `boltz2_aff.ckpt`): the whole pipeline again (5 recycles, 3 samples × 200 steps, confidence) on a 256-token crop, then 2 affinity heads | `boltz2.py:609` | 1× per ligand |

Measured (eager, MPS, warm, stage times synchronized), 136-token protein, 3 recycles, 200 steps, 1 sample:

| Region | Time | Share | Calls |
|---|---:|---:|---:|
| Pairformer (trunk) | 3.71 s | 37% | 4 |
| Score model | 4.50 s | 45% | 200 |
| Sampler glue (not the network) | 1.22 s | 12% | |
| ↳ steering potentials (all return zero) | 0.29 s | | 2,740 |
| ↳ rigid alignment | 0.25 s | | 200 |
| ↳ random augmentation | 0.07 s | | 200 |
| MSA module | 0.30 s | 3% | 4 |
| Template module, confidence, rest | ~0.2 s | 2% | |
| **Total forward** | **9.95 s** | | |

CLI wall time: 54 s for this protein (26 s in the model; the rest is imports, checkpoint
load, featurization, writing), 315 s for the 397-token protein + ligand affinity example.
On a fast GPU the network shrinks much more than the glue does, so the glue's
share grows.

### Data flow (shapes recorded with forward hooks: `dataflow_trace.py`, `reports/dataflow_*.txt`)

136-token protein, 2 samples: N = 136 tokens, M = 1,056 atoms (padded to a multiple of 32),
S = 75 MSA rows, K = M/32 = 33 atom windows.

```
YAML ─parse─▶ processed/*.npz (atoms, residues, chains, bonds; MSA; templates; constraints)
DataLoader (CPU): tokenize (1 token per standard residue, 1 per heavy atom for ligands
                  and modified residues) → featurize → feats: 78 tensors + `record`
   res_type (1,N,33)  msa (1,S,N)  profile (1,N,33)  ref_pos (1,M,3)  atom_to_token (1,M,N)
   token_bonds (1,N,N,1)  contact_conditioning (1,N,N,5)  asym_id, residue_index (1,N) ...

TRUNK (once; 4 passes)
   atom features (1,M,128) + local atom pairs (1,K,32,128,16)
     → windowed atom attention (32 query atoms × 128 key atoms) → mean over atom_to_token
     → s_inputs (1,N,384)   (+ residue type and MSA profile)
   z_init = outer(s_inputs) + rel_pos + bonds + contacts                (1,N,N,128)
   per pass:  z += templates(z);  z += MSA(z, s_inputs)  [MSA rows (1,S,N,64) never leave it]
              s, z = Pairformer×64(s, z)              s (1,N,384), z (1,N,N,128)
   distogram(z)                                        (1,N,N,64)

DIFFUSION CONDITIONING (once)
   (s, z, rel_pos, feats) → q, c (1,M,128); atom biases (1,K,32,128,12) ×2 (3 layers × 4 heads);
   token bias (1,N,N,384) (24 layers × 16 heads); to_keys (66×264 window gather)

SAMPLER (200 steps; the only state carried between steps is the coordinates)
   x (samples,M,3) → center, random rotation/translation, add noise
     → score model: atoms (·,M,128) → 3 atom layers → mean to tokens (·,N,768)
                    → 24 token layers (biases reused) → back to atoms → 3 atom layers → Δx (·,M,3)
     → denoised = c_skip·x + c_out·Δx → align x onto denoised (Kabsch) → Euler step
   → sample_atom_coords (samples,M,3)

CONFIDENCE (once per sample)
   trunk s, z, s_inputs + binned distances between representative atoms of the predicted
   structure (N,N) → Pairformer×8 → pLDDT (N,50 bins), PAE/PDE (N,N,64 bins), resolved
   → pTM, ipTM, chain-pair ipTM → confidence_score = (4·pLDDT + ipTM or pTM)/5 → sample ranking

WRITER (CPU): drop padded atoms → mmCIF/PDB + JSON + npz per sample, in rank order;
   rank-0 sample → predictions/<id>/pre_affinity_<id>.npz when affinity is requested

AFFINITY (boltz2_aff.ckpt, separate weights)
   pre_affinity npz → crop ≤256 tokens / 2,048 atoms around the ligand using the predicted
   coordinates (here 211 tokens, 1,536 atoms) → re-tokenize, re-featurize (adds a single-sequence
   profile, ligand token mask, molecular weight) → trunk (6 passes) → 3 samples × 200 steps
   → confidence → best sample by ipTM
   → z masked to ligand–protein and ligand–ligand pairs (1,211,211,128) + that sample's
     distances → Pairformer-no-seq×8 → mean over those pairs → MLP
   → affinity value and binder probability; two heads averaged
```

The large tensors are pairwise: `z` (N²×128) and the token bias (N²×384). All atom-level work
is windowed, so it is linear in M. The 200 steps share everything except the coordinates and
the noise level, so one step is a self-contained graph once its inputs are all tensors.

What crosses each boundary, which crossings broke capture, and what they are now:

| Boundary | Carries | Upstream problem | Now |
|---|---|---|---|
| featurizer → model | 78 tensors + `record` (Python list) | export needs the tensor-only dict (E9) | the program takes the tensors; `record` stays with the writer |
| trunk → conditioning | `s_inputs`, `s`, `z`, rel-pos (tensors) | none | |
| conditioning → each step | `q`, `c`, 3 bias tensors, **`to_keys` closure** | B6 | indexing-matrix tensor (F2) |
| step → step | coordinates + schedule values pulled to Python floats | B1, B2 | `[200, 4]` scalar tensor, steps run as one scan (F3, F11, F12) |
| recycle → recycle | `s`, `z` | unrolled 4× | one scan (F11) |
| sampler → confidence | coordinates + trunk tensors | none | |
| confidence → output | scores, **chain-pair ipTM as a dict keyed by chain id** | B9–B11 | `(samples, N, N)` tensor; dict rebuilt in `predict_step` (F8–F10) |
| best sample → affinity heads | index taken with `.item()` | B12 | tensor index (F7) |
| structure model → affinity model | **a file on disk**, then CPU re-tokenize, crop, featurize | not a tensor hand-off | tensor crop + gather (`affinity_link.py`), same features |

Inputs used: `inputs/prot.yaml` (136 tokens, 1,056 atoms, 75 MSA rows) and
`inputs/affinity.yaml` (397 tokens, 2,976 atoms, 249 MSA rows; affinity crop 211
tokens). Both use local MSAs; nothing was sent to the MSA server.

## 2. Where the graph breaks (upstream)

Found with a scripted pdb session (`pdb_walk.cmds`, output `reports/pdb_walk_prot.txt`),
`torch._dynamo.explain` and `TORCH_LOGS=graph_breaks` over the full forward, a counting
Dynamo backend, a sync counter, and `torch.export` on each submodule with its real inputs.

### Breaks in the structure model

| # | Location | What happens | Effect |
|---|---|---|---|
| B1 | `modules/diffusionv2.py:372` | `sigma_tm.item(), sigma_t.item(), gamma.item()` every step | 600 syncs. Dynamo: "graph break in loop … skipping the frame", so **the whole `sample()` runs eagerly** and only its callees compile (5+ graphs per step). |
| B2 | `modules/diffusionv2.py:388` → `259` | Score model gets the noise level as a Python float, `torch.full`'d inside | One recompile; constants baked into graphs; not a valid export/CUDA-graph input. (Older torch specialized floats, so each of the 200 values recompiled. That is probably why `Boltz2.__init__` raises `cache_size_limit` to 512.) |
| B3 | `loss/diffusionv2.py:34`, `:57` | `if torch.any(...): print(warning)` in `weighted_rigid_align`, every step | 400 syncs; each alignment splits into 3 graphs. |
| B4 | `potentials/potentials.py:91-96` via `diffusionv2.py:449` | Contact guidance is **on by default** (`BoltzSteeringParams.contact_guidance_update=True`; `--use_potentials` only toggles the other two flags). Each step runs 20 guidance iterations; the contact and template potentials rebuild their constraint lists from `feats` each time and return zeros when there are none | 2,740 tiny graph launches per prediction (73% of all launches), 0.29 s of pure overhead, one recompile (two classes share `compute_gradient`). |
| B5 | `modules/encodersv2.py:64` | `if self.cyclic_pos_enc and torch.any(feats["cyclic_period"] > 0)` (`cyclic_pos_enc=True` in Boltz-2) | Splits the input embedder from the trunk. Inside the confidence loop it makes Dynamo skip the whole confidence `forward`. rel_pos cannot be exported. |
| B6 | `models/boltz2.py:506-527` ← `diffusion_conditioning.py:116` ← `encodersv2.py:350` | `to_keys` is a `functools.partial` closing over a tensor, returned by the conditioning module and fed to the score model each step | Conditioning module cannot be exported ("Found functools.partial in output"; strict: "Reconstruction failure"), and neither can the score model ("Unsupported input type functools.partial"). This is the main missing link between stages. |
| B7 | `layers/triangular_attention/utils.py:87`, `:367` | `type(x) is torch.Tensor` exact-type checks in `chunk_layer` | Non-strict `torch.export` of template, MSA and Pairformer fails (`ValueError: Not supported`): export traces with Tensor subclasses. Strict export works. |
| B8 | `modules/confidencev2.py:121-154` | `run_sequentially`: Python loop over samples calling `forward` recursively, merging dicts | Fine on its own; combined with B5 the whole frame is skipped. |
| B9 | `layers/confidence_utils.py:41-95` | `compute_frame_pred`: per-chain loop over `torch.unique`, `.item()` counts, `if mol_type[...] != NONPOLYMER`, boolean-mask indexing, `try/except` | Data-dependent shapes; frame skipped. |
| B10 | `layers/confidence_utils.py:212-229` | `compute_ptms`: `torch.unique(asym_id).tolist()` and a chain×chain loop building a **nested dict keyed by chain id** | Output structure depends on the data. |
| B11 | `modules/confidencev2.py:480-495` | Bare `try/except Exception` around `compute_ptms` that fills zeros | Dynamo `step_unsupported`. **`torch.export` "succeeds" but the exported program returns pTM = ipTM = 0** (eager pTM 0.176 on the test input; `pair_chains_iptm` turns from a dict into a zeros tensor). Silent wrong confidence scores and sample ranking. |
| B12 | `models/boltz2.py:623` (affinity model) | `best_idx = argsort[0].item()` | Splits the affinity heads from the rest. |

### Other things that matter for export

| # | Location | Issue |
|---|---|---|
| E1 | `layers/triangular_mult.py:7`, `layers/triangular_attention/primitives.py:199` | **CUDA only:** the cuEquivariance kernels (used by default on sm80+ GPUs) are wrapped in `@torch.compiler.disable`. That is 4 breaks per Pairformer layer: about 1,120 in the trunk per prediction (4 passes × 70 layers × 4), plus 32 per confidence sample and 64 in affinity. They need to be `torch.library` custom ops with fake kernels to be traced, and C++-registered ops for AOTInductor. Not testable on this Mac. |
| E2 | `data/module/inferencev2.py:279-285` | `max_tokens=None, max_atoms=None`: every input has its own shapes, so a static graph recompiles per input. |
| E3 | `pairformer.py:181`, `trunkv2.py:595`, `layers/outer_product_mean.py:59`, `triangular_attention/utils.py:320` | Code path changes at 384 tokens (`chunk_size_threshold`), and chunk loops whose trip counts depend on token count and MSA depth. With dynamic shapes these become guards and specializations; with static shapes they unroll (the 397-token input's trunk is 54.5k ops for just 2 passes). |
| E4 | `boltz2.py:440` loop, 64 layers | Unrolled trunk is 57.7k (136 tokens) to about 109k (397 tokens) ops. The first compiled run took 97–153 s with a no-op backend, almost all of it Dynamo tracing; Inductor/AOTI compile time scales with graph size on top of that. |
| E5 | `models/boltz2.py:460-480` | The `compile_*` options wrap modules in `torch.compile`, but `forward` switches back to `_orig_mod` whenever `not self.training`, so they never apply at inference. |
| E6 | `layers/dropout.py:32` | `get_dropout_mask` draws `torch.rand` on the device even in eval (mask of ones): roughly 1,100 wasted random kernels per prediction, and it consumes RNG state. |
| E7 | `models/boltz2.py:353-359` | The checkpoint leaves trunk params `requires_grad=False` but confidence params `True`. Dynamo guards on it and recompiles shared code. Call `model.requires_grad_(False)` before compiling or exporting. |
| E8 | `main.py:1262` | Lightning `precision="bf16-mixed"`: on CUDA the forward runs under bf16 autocast with many `autocast(enabled=False)` islands, so exports must be traced under the same autocast. On MPS Lightning disables autocast (this Mac ran fp32). |
| E9 | feats dict | 78 tensors plus `record` (a list of `Record` objects). Exports need the tensor-only dict. |
| E10 | `trunkv2.py:632` | `torch.randperm` for MSA subsampling runs on the CPU and indexes a device tensor. |

### Side findings (not graph-related)

- `diffusionv2.py:384`: samples are chunked into `multiplicity % max_parallel_samples + 1`
  pieces, so `max_parallel_samples` is mostly ignored. The affinity run asks for 1 at a time
  and gets all 3 at once (3 % 1 + 1 = 1 chunk).
- Featurization applies a random roto-translation to reference conformers with torch's
  global RNG (`featurizerv2.py:1494`). Without `--seed`, two processes see different
  `ref_pos` (up to 8.9 Å apart), so to compare model versions you need a saved batch.
- `_trifast_attn` (`primitives.py:369`) is dead code; the checkpoint's `use_trifast` is ignored.
- The same augmentation loop is `for i in range(torch.max(ref_space_uid))`, so the last
  residue never gets its random rotation. In the affinity example that residue is the ligand:
  its 13 atoms are the only real atoms with identical `ref_pos` in two separate featurizations.

## 3. Fixes on branch `graph-fixes`

Branch `graph-fixes` holds the source changes for sections 3 and 4 as one commit on top of
upstream `b1ebfc4` (`git diff b1ebfc4 graph-fixes`). This report and its scripts are on branch
`graph-audit`, also on top of `b1ebfc4`. F1–F7 alone were six files, +137/−38.

| Fix | Change | Removes |
|---|---|---|
| F1 | `encodersv2.py`: cyclic offset selected with `torch.where(torch.any(is_cyclic), wrapped, d_residue)` | B5 |
| F2 | `diffusion_conditioning.py`, `boltz2.py`, `diffusionv2.py`: conditioning returns `keys_indexing_matrix` (tensor); the score model rebuilds `to_keys` | B6 |
| F3 | `diffusionv2.py`: read the schedule once (`tolist()`), do per-step scalar math on the host in float64 as before, keep a `[steps, 4]` device tensor (t_hat, noise std, step coefficient, `1 / t_hat` taken in float64, see section 7); new `AtomDiffusion.denoise_step(mask, chunks, cond, coords, scalars)` takes only tensors | B1, B2 |
| F4 | `diffusionv2.py`: drop potentials whose constraint list is empty, once per prediction; skip guidance when none are left | B4 |
| F5 | `loss/diffusionv2.py`: remove the two warning `if`s in `weighted_rigid_align` | B3 |
| F6 | `triangular_attention/utils.py`: `isinstance` instead of exact type checks | B7 |
| F7 | `boltz2.py`: affinity best sample via `x[argsort(iptm)[:1]][None]` | B12 |

Steering with real constraints, FK steering and the physical potentials take the old
code path unchanged.

### Verification

Same saved batch, same seed, 3 recycles × 200 steps × 2 samples on MPS: **every output
is bit-identical** to upstream (coords, pLDDT, PAE, PDE, pTM, ipTM, all complex scores,
distogram, `s`, `z`; max |diff| = 0). Upstream run against itself is also bit-identical,
so the comparison is meaningful. The affinity model (5 recycles, 3 samples × 200 steps,
211-token crop, which exercises the new best-sample pick) is bit-identical too, including
`affinity_pred_value` and `affinity_probability_binary`. (On MPS, for the commit before `99c12a6`; from
`99c12a6` the sampler's division follows upstream's CUDA rounding, so on MPS coordinates differ by ≤ 6.7e-4 Å with
every score identical. Section 7.)

### Before / after (136-token protein, 3 recycles, 200 steps, 1 sample)

| | Upstream | `graph-fixes` |
|---|---:|---:|
| Graphs compiled | 21 | 11 |
| Graph launches per prediction | 3,754 | 210 |
| ↳ trunk | 1 graph after a split at rel-pos | 1 graph: embedder → trunk → conditioning (58,103 ops) |
| ↳ diffusion | 200 score + 600 align + 200 aug + 2,740 potentials | 200 (one 2,158-op graph per step) |
| GPU→CPU syncs per prediction | 1,006 | 6 (2 schedule reads, 4 in confidence post-processing) |
| Recompiles | 3 | 0 |
| Breaks left | 20 | the one-time schedule read; confidence pTM/ipTM (B9–B11) |

In the `graph-fixes` column the 200-step loop stays in Python on purpose
(`graph_breaks.py count --sampler-outside`), so each step is one compiled graph. If
Dynamo traces the loop inline it now unrolls it instead: 3 steps already make a
6,483-op graph, and 200 would be about 430k ops.

Timing comparisons on this Mac were not usable: other apps (an MLX Python process,
Java) were using the GPU and CPU, and repeated runs of the same code varied from 13 s to 51 s.
The structural numbers above don't depend on timing. Speed needs measuring on the target GPU.

### Per-module export (real inputs, static shapes)

| Module | Upstream non-strict / strict | `graph-fixes` non-strict / strict |
|---|---|---|
| input_embedder, contact_conditioning, distogram, bfactor | ✅ / ✅ | ✅ / ✅ |
| rel_pos | ❌ data-dependent branch / ❌ | ✅ / ✅ |
| template_module | ❌ `_fetch_dims` / ✅ | ✅ / ✅ |
| msa_module | ❌ `_fetch_dims` / ✅ | ✅ / ✅ |
| pairformer_module (64 layers) | ❌ `_fetch_dims` / ✅ 10.6k nodes | ✅ 11.3k nodes, 32 s / ✅ 58 s |
| diffusion_conditioning | ❌ partial in output / ❌ | ✅ / ✅ |
| score model | ❌ partial input / ❌ | ✅ 1.6k nodes / ✅ |
| confidence_module | ❌ / ❌ | ⚠️ "passes" with pTM/ipTM = 0 (B11) / ❌ |

### Exported pipeline (first version; superseded by section 4, which is one program)

`export_poc.py` exports three programs from the loaded model with
`torch.export.export(strict=False)` and static shapes:

| Program | call_function nodes | Export time |
|---|---:|---:|
| `trunk`: input embedding → 4 recycles (templates, MSA, Pairformer) → distogram | 50,722 | 439–487 s |
| `cond`: diffusion conditioning | 86 | 0.5–4 s |
| `step`: one denoising step (augmentation, noise, score model, alignment, update) | 1,673 | 6–8 s |

Running `trunk → cond → 200 × step` through `ExportedProgram.module()` with the same seed
gives **the same coordinates, `s` and `z` as the eager model (max |diff| = 0)**. The
random ops (MSA subsampling, noise, rotations) are inside the graphs and draw from the
same generators in the same order, which is why the match is exact. Confidence still runs
eagerly afterwards (see B9–B11). These programs are still plain FX graphs: the speed-up
comes from compiling them (Inductor / AOTInductor, a CUDA graph for `step`), which was
not measured here. Wrap only the submodules a program uses (as `export_poc.py` does now):
the first version wrapped the whole model and each saved `.pt2` carried all 2 GB of weights.

## 4. Putting it together: one program

Changes on top of F1–F7, in the same commit (9 files in total, +379/−146):

| Fix | Change | Removes |
|---|---|---|
| F8 | `layers/confidence_utils.py`: `compute_frame_pred_inference` builds the frames of all ligand chains at once (the two nearest atoms in the same chain, via `topk` over an N×N distance matrix), replacing the loop over `torch.unique(asym_id)` with `.item()` counts. The training path keeps the loop | B9 |
| F9 | same file: `compute_ptms` returns chain-pair ipTM as a `(samples, N, N)` tensor over chains in ascending asym id order (one `bmm` with autocast off, then a per-chain max with `scatter_reduce`). `confidencev2.py`: no bare `try/except`; per-sample outputs merge with `torch.cat` | B10, B11 |
| F10 | `boltz2.py` `predict_step`: `confidence_score()` uses `torch.where` instead of `torch.allclose` (no host sync); it also rebuilds the nested chain-pair dict the writer expects | |
| F11 | `modules/utils.py` `repeat_step()`: one `torch._higher_order_ops.scan` when compiling or exporting without autocast on torch 2.15+, otherwise a Python loop. Used for the sampling steps and, at inference, the recycling passes. Eager `scan` would compile its body with Dynamo on every call, which is slow and fails on graph breaks such as the cuEquivariance kernels. Under autocast, scan bodies don't trace (in scan's fake run a bf16 tensor meets float32 weights), so traced loops stay unrolled there. Random draws inside the body happen in the same order either way | the unrolled loops |
| F12 | `diffusionv2.py` `step_schedule()`: the schedule and per-step scalars are read from the device once and cached; a traced forward reuses them as constants | a device read during tracing |
| F13 | `boltz2.py`: the trunk pass builds its own masks. `feats["token_pad_mask"].float()` returns the same tensor (the mask is already float), and scan bodies reject two inputs that share memory | |

**Eager, upstream vs branch.** I used the same saved batch and seed on three inputs:
- the protein (136 tokens, 3 recycles × 200 steps × 2 samples);
- the protein + ligand structure stage (397 tokens, 2 samples), which exercises the new ligand frames;
- the affinity model (211-token crop, 5 recycles, 3 samples).

Every output is bit-identical, including pTM, ipTM, ligand and protein ipTM, and both affinity
outputs. Chain-pair ipTM differs by at most 2.4e-7: its sum now comes from a matrix multiply,
which adds in a different order. Warm eager time is unchanged (protein: 17.3 s vs 17.2 s on MPS). These are MPS
results for the commit before `99c12a6`; section 7 has the CUDA check of `99c12a6`.

### Structure prediction as one program (`export_full.py`)

`Boltz2Program` calls `Boltz2.forward` with the predict-time arguments fixed and returns 15
tensors: coordinates, distogram, pLDDT, PAE, PDE, pTM, ipTM, ligand and protein ipTM,
chain-pair ipTM, the four complex scores and the confidence score. 136-token protein,
3 recycles, 200 steps, 2 samples:

| | |
|---|---|
| `torch.export.export(..., strict=True)` | 92.7 s (the unrolled trunk alone took 439–487 s before) |
| Graph | 4,053 top-level nodes + one trunk pass (11,453 nodes, scan body run 4×) + one sampling step (1,661 nodes, scan body run 200×) |
| Exported vs eager | max abs diff 0 on all 15 outputs |
| vs upstream | same pTM (0.8552, 0.8049) as upstream eager on this batch and seed |

The program covers inputs without steering constraints. `needs_steering(feats)` checks on
the host for contact or pocket constraints and forced templates. Inputs that have them
need the potentials, which are not in the program, so they should run eagerly.

Non-strict export (`strict=False`) fails: it still traces the scan bodies with Dynamo, with
symbolic sizes, and the shape arithmetic in `chunk_layer` then builds an invalid `expand`.
Strict export traces the whole model with Dynamo and works.

### Structure → affinity without the file (`affinity_link.py`)

`crop_mask` is the `AffinityCropper` rule as tensor code:
- distance of each token to the ligand, from the center atoms of the top-ranked sample;
- every token's neighborhood at once: the whole chain if it has ≤ 10 tokens, else the
  smallest residue window with ≥ 10;
- the first step that would exceed 256 tokens, 2,048 atoms or 200 protein tokens.

`crop_and_gather` builds the 34 features the affinity model reads by indexing the
structure-stage features. It also renumbers `token_index` and `ref_space_uid`, remaps frame
atom indices, sets `method_feature` to "other", and builds the single-sequence profile from
MSA row 0.

Checked against the CLI path on the protein + ligand example:

| Check | Result |
|---|---|
| center coordinates from predicted atoms vs the tokenizer's | identical |
| crop vs `AffinityCropper` on the CLI's `pre_affinity` structure | same 211 tokens, 1,521 atoms |
| 34 features vs the CLI featurizer on that crop | identical, except `ref_pos`: the featurizer redraws each residue's random conformer rotation; the gather reuses the structure stage's |
| padded to 256 tokens / 2,048 atoms vs the featurizer with `max_tokens=256, max_atoms=2048` | identical (except `ref_pos`) |
| affinity model (5 recycles, 3 × 200 steps) on the gathered features with the CLI's `ref_pos`, vs on the CLI's features | affinity value, binder probability, coordinates, `z`, ipTM: max abs diff 0 |
| padded vs unpadded crop through the affinity model | `z` on the real tokens identical. Affinity value 2.221 vs 2.283, because the sampler draws noise for 2,048 atoms instead of 1,536 (seeds 0–3: unpadded 2.19–2.32, padded 2.21–2.24; the CLI run 2.246, so the change is the size of a reseed) |

### Structure + affinity as one program (`export_affinity.py`)

`FullProgram` chains: feats → structure program → top-ranked sample (argmax of the confidence
score, the writer's rank 0) → `crop_mask` → `crop_and_gather`, padded to 256 / 2,048 so
shapes are static → affinity model → affinity value and binder probability (both heads).
397-token protein + ligand; structure 3 recycles × 200 steps × 2 samples; affinity
5 recycles × 200 steps × 3 samples:

| | |
|---|---|
| Strict export | 297.6 s |
| Graph | 13,234 top-level nodes. Scan bodies: structure trunk pass 24,825 (397 tokens is past the 384-token chunking branch, so the chunk loops unroll inside it), structure step 1,661, affinity trunk pass 11,099, affinity step 1,661 |
| Exported vs eager | max abs diff 0 on all 21 outputs (15 structure + 6 affinity) |
| Values | affinity 2.104, binder probability 0.343 (the CLI run, with its own random draws: 2.246, 0.340) |

`ExportedProgram.module()` replays the aten graph without compiling it, so the times here
(MPS, noisy: 148 s exported vs 217 s eager for the fused run) say nothing about compiled
speed. Both models' weights are in the program, about 4.3 GB in fp32.

## 5. Still open

1. **Compiled serving.** Section 7 builds and times the AOTInductor package (1.55–2.8× `boltz predict`). Open:
   a GPU snapshot of the loaded package, CUDA graphs for the scan bodies, and a compiled bf16 program.
2. **Shapes.** Each program is specialized to one input's token, atom and MSA counts. The
   model handles padding exactly (the padded trunk `z` was bit-identical on real tokens),
   and the featurizer can pad (`max_tokens`, `max_atoms`). So: pad inputs to a few
   buckets, one program per bucket. Padding changes how many random numbers the sampler
   draws, so samples differ from the unpadded run, as with a different seed. Past 384 tokens
   the chunk loops unroll inside the trunk body (24.8k vs 11.5k nodes).
3. **CUDA kernels (E1).** The cuEquivariance ops are `torch.compiler.disable`d, which strict
   export cannot trace. They are already `torch.library` custom ops with fake kernels: without the decorators
   they export and match eager (section 7). AOTInductor with them is blocked by the same compiler crash.
4. **Precision (E8).** Lightning runs Boltz-2 under bf16 autocast, on CUDA and on CPU. A program
   traced under autocast keeps the recycling loop unrolled (F11); the sampler runs with
   autocast off and stays one scan. Checked on CPU: a bf16-autocast strict export (2 recycling
   passes, 2 steps) took 679 s and matches eager exactly. On CUDA the export traces but does not replay: torch
   2.15 records wrong dtypes around `autocast(enabled=False)` blocks (section 7).
5. **Steering with constraints.** Inputs with contacts, pockets or forced templates still
   need the eager sampler. A scanned version needs the per-step potential parameters as
   tensors, and the template potential's boolean-mask indexing replaced.
6. **Trunk body size.** The trunk pass still unrolls 64 Pairformer layers (11.5k nodes). A scan
   over stacked layer weights would make it about 64× smaller and cut compile time.
7. **Non-strict export** fails inside scan bodies (see above); strict works. The scan path is
   enabled on torch 2.15 or newer, the version tested; older versions trace unrolled loops, and
   eager runs keep Boltz's `torch>=2.2`. On 2.15, Inductor (`torch.compile`) also compiles the
   scans; its scan pass needs a tensor per-step output, so the body returns one.

Done since the first version of this list: confidence post-processing, the sampler as one
graph, the recycling unroll, and the affinity hand-off.

## 6. Reproduce

The commands assume the layout this work was done in: upstream code in the repository root
(the "upstream" side) with these scripts in `graph_audit/`, which is branch `graph-audit`, and
branch `graph-fixes` as a worktree at `graph_audit/wt-fixes` (the patched side):

```bash
git clone <this repository> boltz && cd boltz
git checkout graph-audit                          # upstream b1ebfc4 + graph_audit/
pip install -e .                                  # Boltz and its dependencies (weights go to ~/.boltz)
git worktree add graph_audit/wt-fixes graph-fixes
cd graph_audit
```

Large outputs (`*.pt`: saved batches and predictions) and the CLI run directories are not in the
repository; the commands below regenerate them.

```bash
cd ~/Desktop/boltz && source .venv/bin/activate && cd graph_audit
boltz predict inputs/prot.yaml --out_dir runs/cli_prot --accelerator gpu --override   # featurize once
R=runs/cli_prot/boltz_results_prot
python -m pdb walk.py $R < pdb_walk.cmds                     # guided pdb tour of the breaks
TORCH_LOGS=graph_breaks python graph_breaks.py explain --run $R --recycles 1 --steps 3 --samples 2 --out reports/explain_prot.json
python graph_breaks.py count  --run $R --recycles 3 --out reports/count_prot.json
python graph_breaks.py modules --run $R --out reports/modules_prot.json
python sync_points.py --run $R --out reports/sync_prot.json
python profile_stages.py --run $R --out reports/stages_prot.json
# fixed branch: prefix with PYTHONPATH=$PWD/wt-fixes/src (add --sampler-outside to count)
python compare.py save --run $R --samples 2 --out reports/out_orig.pt
PYTHONPATH=$PWD/wt-fixes/src python compare.py save --run $R --samples 2 --out reports/out_fixed.pt
python compare.py diff reports/out_orig.pt reports/out_fixed.pt
PYTHONPATH=$PWD/wt-fixes/src python export_poc.py --batch reports/batch_boltz_results_prot.pt
```

Section 4:

```bash
P=$PWD/wt-fixes/src
PYTHONPATH=$P python export_full.py --strict --batch reports/batch_boltz_results_prot.pt --out reports/export_full_prot.json
python compare.py save --run runs/cli_aff/boltz_results_affinity --samples 2 --out reports/v2_ligstruct_orig.pt   # caches the structure-stage batch
PYTHONPATH=$P:$PWD python scratch/link_check.py        # crop + features vs the CLI path (CPU); writes scratch/link_feats.pt
PYTHONPATH=$P python export_affinity.py                # link / padding / fused export checks
```

Files: `harness.py` (loads model and batch like the CLI), `walk.py` + `pdb_walk.cmds`, `dataflow_trace.py`,
`graph_breaks.py`, `sync_points.py`, `profile_stages.py`, `compare.py`, `timing.py`,
`export_poc.py`, `export_full.py`, `affinity_link.py`, `export_affinity.py`,
`scratch/link_check.py`; results in `reports/`.

## 7. On a CUDA GPU (Modal)

PR #1 (commit `74e4526`; `99c12a6` after the change below), installed from the fork next to upstream `b1ebfc4` in one image: torch 2.15.0
(CUDA 13.2), Triton 3.9, cuEquivariance 0.13, g++ 13.3. GPU: NVIDIA L40S (sm 8.9); the debugging runs used an
A10 (sm 8.6) while no L40S was free. Same saved batch and seed as section 4 (136-token protein,
3 recycles × 200 steps × 2 samples), fp32 unless noted. Code: `cuda_modal.py` (image, Volume, GPU functions,
snapshot workers, a CPU shell with gdb), `cuda_bench.py` (the stages); results in `reports/cuda_*.json`.

| Question | Answer |
|---|---|
| Strict export on CUDA | Works. Same graph as on MPS (2 scans; trunk body 11,454 nodes, step body 1,662). Exported vs eager: max abs diff 0 on all 15 outputs |
| PR #1 eager vs upstream on CUDA | `74e4526`: sampler off by one rounding per step (coordinates ≤ 3.3e-4 Å after 200 steps, pTM ≤ 5.8e-5), traced with pdb to the `x / t_hat` update. `99c12a6` follows upstream's rounding: every output identical (chain-pair ipTM 1.2e-7) |
| AOTInductor | Works after two build fixes, both found with gdb: clang++ instead of g++ (stack overflow) and `-march=x86-64-v3` instead of native (SIGILL on other hosts). One prediction: 3.19 s fp32 and 1.78 s with TF32 matmuls, vs 4.95 s for what `boltz predict` runs (1.55× and 2.8×) |
| cuEquivariance kernels | Export once Boltz's `torch.compiler.disable` wrappers are removed: 3 custom ops in the graph, exact vs eager. AOTInductor with them hit the g++ crash and was not retried with clang |
| bf16 autocast | The export traces but the program doesn't replay: torch 2.15 records wrong dtypes around `autocast(enabled=False)` blocks |
| GPU memory snapshot | Eager `boltz predict` worker: cold start 56–60 s → 7.5–14 s after restore, outputs identical |

### Export and parity (`74e4526`, L40S)

| Run | Result |
|---|---|
| Upstream eager, twice | bit-identical (CUDA kernels deterministic here) |
| Strict export (`export_full.py --strict`) | 256 s (Modal's CPU; 93 s on the M4 Max) |
| Exported program replayed vs eager | max abs diff 0 on all 15 outputs |
| Eager time (fp32, warm) | upstream 9.9–10.9 s, PR #1 9.5 s, exported (not compiled) 9.3 s |

**Why `74e4526` was not bit-identical to upstream on CUDA** (it was on MPS). Traced with pdb breakpoints on the update
line of both versions, saving the operands at every step (`cuda_bench.py steptrace`):

- Step 0: the score network's input, noise level and output are bit-identical, and so are the update's operands.
  The update's result is not.
- The update divides by `t_hat`. Upstream passes a Python float; on CUDA, PyTorch computes `x / s` for a host scalar
  as `x * (1/s)`, with `1/s` taken in float64. PR #1 divides by a float32 device tensor, which is true division.
  On CPU and MPS both are true division, which is why MPS matched.
- Proof, 200 steps on CUDA: writing the update as `x * inv_t_hat`, with `inv_t_hat = 1 / t_hat` computed in float64
  as a fourth column of `step_schedule`, gives **0 differing steps** and final outputs identical to upstream (chain-pair
  ipTM 1.2e-7, the summation order from section 4). `x * (1 / t_hat)` with a float32 reciprocal is identical for
  steps 0–5 and first differs at step 6, where the two reciprocals round differently.
- The same change makes MPS differ from upstream instead (coordinates 6.7e-4 Å). Upstream's division itself rounds
  differently per device, so PR #1 can match it on CUDA or on CPU/MPS, not both.
- PR #1 now takes the CUDA side (`99c12a6`: a fourth `step_schedule` column, `x * inv_t_hat`). Checked with that
  commit on CUDA (A10, `reports/cuda_parity.json`): upstream vs PR #1 identical on every output except chain-pair
  ipTM (1.2e-7); upstream twice identical; strict export 216 s, 2 scans, exported vs eager max abs diff 0. On MPS:
  coordinates ≤ 6.7e-4 Å, every score, distogram, `s` and `z` identical; exported vs eager still 0. The L40S parity
  run of `74e4526` is `reports/cuda_parity_l40s_74e4526.json`.

### AOTInductor

Four problems, all in the toolchain, not in Boltz:

1. With `fallback_random=True` (compiled code calls eager's random ops, to compare against eager), Inductor wraps
   the random ops in `control_deps` to keep their order, and its `decompose_scan_to_while_loop` pass then can't
   re-trace the scan bodies (`NotImplementedError: no rule registered for HigherOrderOperator control_deps`).
2. Default settings: lowering and Triton compilation finish (about 15 minutes), then g++ crashes on the wrapper
   (`internal compiler error: Segmentation fault`), also at `-O0`. Traced with gdb on `cc1plus`
   (`scratch/gxx_gdb.sh`): the fault address is just below the `[stack]` mapping, which is 8 MiB even though
   `ulimit -s` says unlimited (Modal's sandbox), and the backtrace is 53,706 frames of `walk_tree_1` ↔
   `c_genericize_control_stmt`, GCC recursing through the wrapper's nested statements. With a finite 1 GiB limit
   (`ulimit -s 1048576`) g++ gets past the crash but was still compiling after 55 minutes, when I stopped it.
   clang++ 18 compiles the same file in 322 s with a 1.4 GB peak (`scratch/clang_try.sh`), so the image sets
   `CXX=clang++` (plus `libomp-dev` for Inductor's CPU kernels).
3. Once built, the package died with SIGILL in the next container. gdb (`scratch/sigill_gdb.sh`) stops at
   `vcvtusi2sd %rcx,%xmm0,%xmm0`, an AVX-512F instruction in a `std::unordered_map` rehash inlined into
   `wrapper.so`, reached from `AOTInductorModelBase::load_constants`. Inductor builds with `-march=native`; the
   compile host had AVX-512, that one only AVX2. A scan for `zmm` registers finds none, because the instruction
   uses `xmm`. Modal mixes CPU types under one GPU type, so the image sets `TORCHINDUCTOR_CPP_MARCH=x86-64-v3`.
4. `torch.export.save` can't serialize the program (`symbol '_enter_autocast_6' not in def_table`), so it is
   exported and compiled in one process; the AOTInductor package is the artifact.

Built that way (clang++, `x86-64-v3`, wrapper at `-O0`): compile 754–1,188 s on top of a 173 s export, a 2.07 GB
package, 1.9 s to load. Two runs with the same seed are bit-identical. Against eager, the trunk matches to fusion
rounding (distogram 8.4e-5). The samples differ, because Inductor draws its own random numbers, but land in the
same range (pTM 0.81 / 0.75 vs 0.83 / 0.76, pLDDT 0.82 / 0.79 vs 0.83 / 0.79). The `-9223372036854775808L` literal
g++ warns about (Inductor's seed `randint` inside the loop body) is only a warning.

### Speed (L40S, one container, warm median of 3)

| Variant | One prediction | vs `boltz predict` |
|---|---:|---:|
| eager fp32 | 4.91 s | |
| eager fp32, TF32 matmuls | 4.70 s | |
| eager bf16-mixed | 4.74 s | |
| eager bf16-mixed + cuEquivariance (what `boltz predict` runs) | 4.95 s (first call 10.3 s) | 1.0× |
| AOTInductor fp32 | **3.19 s** | 1.55× |
| AOTInductor fp32, TF32 matmuls | **1.78 s** | 2.8× |

Eager costs the same at every precision: it is bound by kernel launches and Python, not by GPU arithmetic, and
across containers the same eager run took 4.9–10.5 s with the host CPU. The compiled program launches from C++ and
fuses the glue around the network. TF32 changes the samples as much as a different seed (pTM 0.78 / 0.75 vs 0.81 /
0.75), the same as bf16 does in eager. The package was built in one container and timed in another.

### cuEquivariance kernels (`use_kernels=True`, 1 recycle, 3 steps)

- As is: strict export fails, `UncapturedHigherOrderOpError ... Skip calling torch.compiler.disable()d function`
  (`kernel_triangular_mult` inside the trunk scan body; E1).
- With the decorators removed (patched in the test only): export takes 167–234 s and the graph calls
  `cuequivariance.triangle_attention`, `fused_gated_dual_gemm` and `layer_norm_transpose`, which are Python
  `torch.library` custom ops. Exported vs eager: max abs diff 0. AOTInductor: the same g++ stack overflow.
- Eager with the kernels, this small config: 7.1 s fp32, 2.8 s bf16 (first call, includes kernel set-up).

### bf16 autocast

Lightning's bf16-mixed, exported on CUDA (1 trunk pass, 2 steps; the trunk unrolls under autocast, F11). Export
works (524 s, or 157 s with autocast inside the module) but replay fails on torch's dtype checks. pdb post-mortem
(`cuda_bench.py bf16debug`):

- Autocast around `torch.export.export`: the failing check is `to_keys(atom_mask.unsqueeze(-1).float())`
  (`encodersv2.py:366`), inside the atom encoder's `autocast(enabled=False)` block. Export recorded its einsum as
  bf16, as if the outer autocast applied there; at replay the block runs in fp32. Replaying with or without an
  outer autocast fails the same way. (On CPU, section 5 item 4, autocast covers fewer ops, so this passed.)
- Autocast inside the exported module: the reverse. Ops before the atom transformer's `enabled=False` block were
  recorded as fp32, but replay in bf16. The trunk loop also became a scan, so `torch.is_autocast_enabled()` read
  the global state, not the traced block.

### Cold start with GPU memory snapshots

Workers run what `boltz predict` runs on an sm80+ GPU: eager, bf16-mixed, cuEquivariance kernels. Every call gets
a new container. Cold penalty = client latency (from Korea) − prediction time.

| Worker | Runs | Cold penalty | In the container |
|---|---:|---|---|
| No snapshot | 2 | 59.7 s, 56.2 s | imports 2–3 s, model load 30–31 s, first (warm-up) run 17 s |
| Snapshot being created | 2 | 66.6 s, 373.7 s (queued for a GPU) | as above, then the snapshot |
| Restored from the snapshot | 2 | 14.4 s, 7.5 s | first GPU op after restore 0.004–0.1 s |

All six returned identical outputs, equal to the pre-snapshot warm-up. Prediction time varied 6.0–11.6 s between
containers on the same GPU type: the eager sampler is launch-bound, so host CPU speed shows. The AOTInductor
worker (`AOTISnapshot`, `AOTIBaseline` in `cuda_modal.py`) was not measured: the Modal spend limit was close.

### Cost

$11.24 on Modal for everything in this section, about two thirds of it L40S time (covered by credits; billed $0).

### Commands

```bash
M=../.venv-modal/bin/modal                                   # from graph_audit/; pip install modal
$M run cuda_modal.py::setup                                  # weights into the Volume (CPU)
$M run cuda_modal.py::main --stage parity                    # then aoti, timing, kernels, bf16debug
$M run cuda_modal.py::main --stage steptrace --gpu A10G      # the pdb trace of the sampler update
$M run cuda_modal.py::debug --script scratch/gxx_gdb.sh      # g++ under gdb, CPU only
$M run cuda_modal.py::debug --script scratch/sigill_gdb.sh --gpu L40S   # the package under gdb
$M deploy cuda_modal.py && ../.venv-modal/bin/python cuda_modal.py coldstart EagerBaseline EagerSnapshot
$M app stop boltz-export-cuda --yes && $M volume delete boltz-export --yes
```
