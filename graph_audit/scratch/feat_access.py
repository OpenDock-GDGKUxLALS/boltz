"""Which feats keys does a forward read? (affinity model, tiny settings, CPU)"""
import sys
import torch
from harness import load_model, run_forward


class Tracking(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read = set()

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def get(self, k, d=None):
        self.read.add(k)
        return super().get(k, d)

    def __contains__(self, k):
        self.read.add(k)
        return super().__contains__(k)


affinity = sys.argv[1] == "aff"
batch = torch.load("reports/batch_boltz_results_affinity_aff.pt" if affinity else "reports/batch_boltz_results_prot.pt")
if affinity:
    batch["affinity_mw"] = [torch.tensor(400.0)]
torch.set_grad_enabled(False)
model = load_model(torch.device("cpu"), affinity=affinity, recycling_steps=0, sampling_steps=2, diffusion_samples=1)
feats = Tracking(batch)
out = run_forward(model, feats)
print(sorted(feats.read))
print("unused:", sorted(set(batch) - feats.read))
