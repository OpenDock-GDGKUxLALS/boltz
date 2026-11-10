import torch
from torch._higher_order_ops.scan import scan


class M(torch.nn.Module):
    def forward(self, z, bias):
        def body(c, _):
            dims = tuple(max(s) for s in zip(c.shape[:2], bias.shape[:2]))
            return c + bias.expand(dims + bias.shape[2:]).sum() * 0 + 1, None
        out, _ = scan(body, z, None, length=2)
        return out


z = torch.zeros(1, 7, 7, 4)
bias = torch.zeros(1, 1, 7, 4)
for strict in (False, True):
    print("strict", strict)
    try:
        torch.export.export(M(), (z, bias), strict=strict)
        print("  ok")
    except Exception as e:
        print("  FAIL", str(e).splitlines()[0][:200])
