import torch
from torch._higher_order_ops.scan import scan


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)
        self.__dict__["_cache"] = {}

    def table(self):
        if "t" not in self._cache:
            self._cache["t"] = torch.rand(4, 2, device=self.lin.weight.device)
        return self._cache["t"]

    def forward(self, x):
        t = self.table()
        y0 = t[0, 0] * x
        y, _ = scan(lambda c, s: (self.lin(c) * s[1], None), y0, t)
        return y


for dev in ("cpu", "mps"):
    m = M().to(dev)
    x = torch.randn(2, 3, device=dev)
    m.table()
    with torch.no_grad():
        ref = m(x)
        try:
            ep = torch.export.export(m, (x,), strict=False)
            print(dev, "ok", torch.equal(ep.module()(x), ref), [s.kind.name for s in ep.graph_signature.input_specs])
        except Exception as e:
            print(dev, "FAIL", type(e).__name__, str(e)[:400])
