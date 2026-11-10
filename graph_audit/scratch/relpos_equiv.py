"""RelativePositionEncoder: branch vs upstream, batch size 1, with and without a cyclic chain."""
import importlib.util
import sys
import torch

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

up, new = load(sys.argv[1], "up_enc"), load(sys.argv[2], "new_enc")
N = 40
for label, period in [("no cyclic", torch.zeros(1, N)), ("cyclic chain", torch.cat([torch.full((1, 25), 25.0), torch.zeros(1, 15)], 1))]:
    torch.manual_seed(0); a = up.RelativePositionEncoder(token_z=16, cyclic_pos_enc=True)
    b = new.RelativePositionEncoder(token_z=16, cyclic_pos_enc=True); b.load_state_dict(a.state_dict())
    asym = torch.cat([torch.zeros(1, 25), torch.ones(1, 15)], 1).long()
    feats = {"residue_index": torch.cat([torch.arange(25), torch.arange(15)])[None], "cyclic_period": period,
             "asym_id": asym, "entity_id": asym, "sym_id": torch.zeros(1, N, dtype=torch.long),
             "token_index": torch.arange(N)[None], "mol_type": torch.zeros(1, N, dtype=torch.long)}
    print(f"{label:12s}: identical to upstream = {torch.equal(a(feats), b(feats))}")
