"""repeat_step under torch.compile with Inductor (fullgraph), CPU: scan path and loop fallback."""
import torch

import boltz.model.modules.utils as u

w = torch.randn(8, 8)
step = lambda c, x: torch.tanh(c @ w + x)  # noqa: E731
c, xs = torch.randn(2, 8), torch.randn(5, 2, 8)
ref = u.repeat_step(step, c, xs)
for label, scan in (("scan", u.scan), ("loop fallback", None)):
    u.scan = scan
    out = torch.compile(lambda c, xs: u.repeat_step(step, c, xs), fullgraph=True)(c, xs)
    print(f"inductor {label}: max|d| = {(out - ref).abs().max().item()}")
