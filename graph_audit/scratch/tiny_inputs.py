"""compute_frame_pred(inference=True) on inputs with fewer than 3 tokens, vs upstream;
and repeat_step under torch.compile when torch has no scan."""
import importlib.util
import sys

import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


up, new = load(sys.argv[1], "up_cu"), load(sys.argv[2], "new_cu")
for label, mol_type, atoms in [("2-residue protein", [0, 0], [3, 3]), ("single ion", [3], [1]),
                               ("2-atom ligand", [3, 3], [1, 1])]:
    n, m = len(mol_type), sum(atoms)
    tok = torch.repeat_interleave(torch.arange(n), torch.tensor(atoms))
    feats = {"asym_id": torch.zeros(1, n, dtype=torch.long), "mol_type": torch.tensor([mol_type]),
             "token_pad_mask": torch.ones(1, n), "atom_pad_mask": torch.ones(1, m),
             "atom_to_token": torch.nn.functional.one_hot(tok, n)[None]}
    first = torch.cumsum(torch.tensor([0] + atoms[:-1]), 0)
    frames = torch.stack([first, first + (torch.tensor(atoms) > 1), first + 2 * (torch.tensor(atoms) > 2)], -1)[None, None]
    coords = torch.randn(2, m, 3)
    try:
        a = up.compute_frame_pred(coords, frames, feats, 2, inference=True)
    except Exception as e:  # noqa: BLE001
        a = e
    b = new.compute_frame_pred(coords, frames, feats, 2, inference=True)
    same = not isinstance(a, Exception) and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    print(f"{label:18s}: upstream {'ok' if not isinstance(a, Exception) else 'FAILS'}, new ok, identical={same}")

import boltz.model.modules.utils as u  # noqa: E402

u.scan = None  # as on torch < 2.6
f = torch.compile(lambda c: u.repeat_step(lambda c, x: c + x, c, torch.ones(3, 2)), backend="eager", fullgraph=True)
print("compile without scan:", f(torch.zeros(2)).tolist())
