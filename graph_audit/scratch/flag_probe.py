import torch
seen = []


class M(torch.nn.Module):
    def forward(self, x):
        seen.append((torch.compiler.is_compiling(), torch.compiler.is_exporting(), torch.compiler.is_dynamo_compiling()))
        return x + 1


x = torch.ones(2)
M()(x); print("eager       (compiling, exporting, dynamo):", seen[-1])
torch.export.export(M(), (x,), strict=False); print("export ns   :", seen[-1])
torch.export.export(M(), (x,), strict=True); print("export strict:", seen[-1])
torch.compile(M(), backend="eager")(x); print("compile     :", seen[-1])
