"""Probe torch 2.15 scan / while_loop under torch.export: closures, params, RNG in body."""
import torch
from torch import nn
from torch._higher_order_ops.scan import scan
from torch._higher_order_ops.while_loop import while_loop

torch.manual_seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x, t):
        return self.lin(x) * t


class ScanLoop(nn.Module):
    def __init__(self, rng_in_body=False):
        super().__init__()
        self.net = Net()
        self.rng_in_body = rng_in_body

    def forward(self, x0, sched, cond, noise):
        def body(x, xs):
            s, e = xs
            if self.rng_in_body:
                e = torch.randn_like(x)
            y = x + s[0] * e
            y = self.net(y, s[1]) + cond  # closure over module params and a graph tensor
            return y.clone(), y.sum().reshape(1)

        xf, ys = scan(body, x0, (sched, noise))
        return xf, ys


class WhileLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Net()

    def forward(self, x0, sched, cond, noise):
        n = sched.shape[0]

        def c(i, x):
            return i < n

        def b(i, x):
            s = sched[i]
            e = noise[i]
            y = x + s[0] * e
            y = self.net(y, s[1]) + cond
            return i + 1, y.clone()

        _, xf = while_loop(c, b, (torch.zeros((), dtype=torch.int64, device=x0.device), x0))
        return xf


def eager_ref(m, x0, sched, cond, noise):
    x = x0
    for k in range(sched.shape[0]):
        x = m.net(x + sched[k, 0] * noise[k], sched[k, 1]) + cond
    return x


S = 5
x0 = torch.randn(2, 7, 3, device=dev)
sched = torch.rand(S, 2, device=dev)
cond = torch.randn(2, 7, 3, device=dev)
noise = torch.randn(S, 2, 7, 3, device=dev)

for name, mod in [("scan", ScanLoop().to(dev)), ("while", WhileLoop().to(dev)), ("scan+rng", ScanLoop(True).to(dev))]:
    mod.eval()
    for strict in (False, True):
        try:
            with torch.no_grad():
                ep = torch.export.export(mod, (x0, sched, cond, noise), strict=strict)
                out = ep.module()(x0, sched, cond, noise)
                xf = out[0] if isinstance(out, (tuple, list)) else out
                ref = eager_ref(mod, x0, sched, cond, noise)
                hops = [n.target for n in ep.graph.nodes if n.op == "call_function" and "higher_order" in str(n.target)]
                print(f"{name:9s} strict={strict}: OK  max|d|={(xf - ref).abs().max().item():.3g}  hops={hops}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:9s} strict={strict}: FAIL {type(e).__name__}: {str(e).splitlines()[0][:300]}")
