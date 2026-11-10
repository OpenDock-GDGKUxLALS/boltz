"""Reproduce CodeRabbit's two findings against the current branch code."""
import torch

from boltz.model.modules.encodersv2 import RelativePositionEncoder
from boltz.model.modules.utils import repeat_step

# 1. batch of 2, no cyclic chains
enc = RelativePositionEncoder(token_z=8, cyclic_pos_enc=True)
B, N = 2, 3
feats = {"residue_index": torch.arange(N).repeat(B, 1), "cyclic_period": torch.zeros(B, N),
         "asym_id": torch.zeros(B, N, dtype=torch.long), "entity_id": torch.zeros(B, N, dtype=torch.long),
         "sym_id": torch.zeros(B, N, dtype=torch.long), "token_index": torch.arange(N).repeat(B, 1),
         "mol_type": torch.zeros(B, N, dtype=torch.long)}
try:
    enc(feats)
    print("1. B=2 rel_pos: ok")
except Exception as e:  # noqa: BLE001
    print("1. B=2 rel_pos: FAILS:", type(e).__name__, str(e).splitlines()[0][:120])

# 2. repeat_step(length=0) under torch.compile
f = torch.compile(lambda c: repeat_step(lambda c, _: c * 2, c, torch.empty(0, 0)), backend="eager", fullgraph=True)
try:
    print("2. compiled length=0:", f(torch.ones(2)).tolist())
except Exception as e:  # noqa: BLE001
    print("2. compiled length=0: FAILS:", type(e).__name__, str(e).splitlines()[0][:120])
