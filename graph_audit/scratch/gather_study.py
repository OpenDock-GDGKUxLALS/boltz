import torch
import torch.nn.functional as F

full = torch.load("reports/batch_boltz_results_affinity.pt")
aff = torch.load("reports/batch_boltz_results_affinity_aff.pt")
s = torch.load("scratch/crop_sel.pt")
t, a = s["token_idx"], s["atom_idx"]
N, M = full["res_type"].shape[1], full["atom_pad_mask"].shape[1]
Mc = aff["atom_pad_mask"].shape[1]
pad_a = Mc - len(a)
old2new = torch.full((M,), -1, dtype=torch.long)
old2new[a] = torch.arange(len(a))


def pad_atoms(x, dim):
    shape = list(x.shape); shape[dim] = pad_a
    return torch.cat([x, torch.zeros(shape, dtype=x.dtype)], dim)


def gather(k, x):
    sh = x.shape
    if k in ("atom_to_token",):
        return pad_atoms(x[:, a][:, :, t], 1)
    if k in ("token_to_rep_atom", "token_to_center_atom"):
        return pad_atoms(x[:, t][:, :, a], 2)
    if k == "frames_idx":
        return old2new[x[:, :, t]]
    if k in ("msa", "msa_mask", "msa_paired", "deletion_value", "has_deletion"):
        return x[:, :, t]
    if len(sh) >= 3 and sh[1] == N and sh[2] == N:
        return x[:, t][:, :, t]
    if len(sh) >= 2 and sh[1] == N:
        return x[:, t]
    if len(sh) >= 3 and sh[1] == 1 and sh[2] == N:
        return x[:, :, t]
    if len(sh) >= 2 and sh[1] == M:
        return pad_atoms(x[:, a], 1)
    if len(sh) >= 3 and sh[1] == 1 and sh[2] == M:
        return pad_atoms(x[:, :, a], 2)
    return None


rows = []
for k in sorted(aff):
    y = aff[k]
    if k == "profile_affinity":
        g = F.one_hot(full["msa"][0, 0, t], 33).float()[None]
    elif k == "deletion_mean_affinity":
        g = full["deletion_value"][0, 0, t][None]
    else:
        g = gather(k, full[k]) if k in full else None
    if g is None:
        rows.append((k, "no rule", tuple(full[k].shape) if k in full else None, tuple(y.shape)))
        continue
    if g.shape != y.shape:
        rows.append((k, "shape", tuple(g.shape), tuple(y.shape)))
        continue
    if torch.equal(g, y):
        rows.append((k, "equal", "", ""))
    else:
        d = (g.float() - y.float()).abs()
        rows.append((k, "DIFF", f"max|d|={d.max().item():.4g}", f"n_diff={(d > 0).sum().item()}/{d.numel()}"))
for r in rows:
    print(f"{r[0]:28s} {r[1]:8s} {r[2]!s:22s} {r[3]!s}")
