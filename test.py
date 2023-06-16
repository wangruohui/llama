import torch
import torch.nn as nn

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


norm = RMSNorm(2048)
norm2 = torch.compile(norm,  backend="inductor")

x = torch.randn(2048)

with torch.inference_mode():
    print(norm(x))
    print(norm2(x))

    def timed(fn, x):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(x)
        end.record()
        torch.cuda.synchronize()
        return result, start.elapsed_time(end) / 1000

    x = torch.randn(2048)
    r, t1 = timed(norm,x)
    r, t2 = timed(norm2,x)
    print(r, t1)
    print(r, t2)


from torch.profiler import profile, record_function, ProfilerActivity