"""Does Inductor (torch.compile default backend) compile a scan on this torch? CPU."""
import torch
from torch._higher_order_ops.scan import scan

w = torch.randn(8, 8)


def f(c, xs):
    return scan(lambda c, x: (torch.tanh(c @ w + x).clone(), x.clone()), c, xs)[0]


c, xs = torch.randn(2, 8), torch.randn(5, 2, 8)
ref = f(c, xs)
out = torch.compile(f, fullgraph=True)(c, xs)
print(torch.__version__, "inductor scan ok, max|d| =", (out - ref).abs().max().item())
