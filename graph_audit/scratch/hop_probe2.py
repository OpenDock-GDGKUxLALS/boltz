import torch
from torch._higher_order_ops.scan import scan

dev = "mps"
print("public torch.scan:", hasattr(torch, "scan"))


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)

    def body(self, x, s):
        q = torch.randn((2, 4), device=x.device)
        e = s[0] * torch.randn(x.shape, device=x.device)
        idx = torch.randperm(x.shape[1])[:5]           # CPU generator, like the MSA subsampling
        y = self.lin(x + e + q.sum()) * s[1]
        y = y + y[:, idx].mean()
        return y

    def forward(self, x, sched, use_scan: bool = True, ys_none: bool = True):
        if not use_scan:
            for k in range(sched.shape[0]):
                x = self.body(x, sched[k])
            return x

        def comb(c, s):
            return self.body(c, s), (None if ys_none else s.sum().reshape(1))
        x, _ = scan(comb, x, sched)
        return x


m = M().to(dev).eval()
x0 = torch.randn(2, 7, 3, device=dev)
sched = torch.rand(6, 2, device=dev)
with torch.no_grad():
    torch.manual_seed(1); ref = m(x0, sched, use_scan=False)
    for ys_none in (True, False):
        try:
            torch.manual_seed(1); eag = m(x0, sched, True, ys_none)
            ep = torch.export.export(m, (x0, sched), kwargs={"use_scan": True, "ys_none": ys_none}, strict=False)
            torch.manual_seed(1); exp = ep.module()(x0, sched, use_scan=True, ys_none=ys_none)
            print(f"ys_none={ys_none}: eager-scan vs loop {(eag-ref).abs().max().item()}  exported vs loop {(exp-ref).abs().max().item()}")
        except Exception as e:
            print(f"ys_none={ys_none}: FAIL {type(e).__name__}: {str(e)[:300]}")

# scatter_reduce amax on MPS + export
v = torch.rand(1, 9, 4, device=dev)
ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 0]], device=dev)
out = torch.zeros_like(v).scatter_reduce(1, ids[..., None].expand_as(v), v, reduce="amax", include_self=True)
ref = torch.stack([v[0][ids[0] == c].max(0).values for c in range(3)])
print("scatter amax mps ok:", torch.equal(out[0, :3], ref), out.shape)


class S(torch.nn.Module):
    def forward(self, v, ids):
        return torch.zeros_like(v).scatter_reduce(1, ids[..., None].expand_as(v), v, reduce="amax", include_self=True)


ep = torch.export.export(S(), (v, ids), strict=False)
print("scatter export ok:", torch.equal(ep.module()(v, ids), out))
# topk on mps
d = torch.rand(1, 1, 50, 50, device=dev)
print("topk vs sort:", torch.equal(torch.topk(d, 3, dim=-1, largest=False).indices, torch.sort(d, dim=-1).indices[..., :3]))
