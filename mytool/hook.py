from dataclasses import dataclass

import torch


@dataclass
class Stat:
    min: float = 0
    max: float = 0
    mean: float = 0
    M2: float = 0
    count: int = 0

    def update(self, value):
        assert value.ndim == 2
        local_count = value.shape[0]
        local_min = value.amin(dim=0).to(torch.float64)
        local_max = value.amax(dim=0).to(torch.float64)
        local_mean = value.mean(dim=0).to(torch.float64)
        local_M2 = value.var(dim=0, correction=0).to(torch.float64) * local_count

        if self.count == 0:
            self.min, self.max = local_min, local_max
            self.mean, self.M2 = local_mean, local_M2
        else:
            self.min = torch.minimum(self.min, local_min)
            self.max = torch.maximum(self.max, local_max)
            delta = local_mean - self.mean
            new_count = self.count + local_count
            self.M2 = (
                self.M2 + local_M2 + delta**2 * self.count * local_count / new_count
            )
            self.mean = self.mean + delta * local_count / new_count

        self.count += local_count

    @property
    def std(self):
        return torch.sqrt(self.M2 / self.count)

    def __str__(self):
        return f"min: {self.min}\nmax: {self.max}\nmean: {self.mean}\nstd: {self.std}\ncount: {self.count}"


class PerChannelStatHook:
    stat = {}

    def __init__(self, name, which="output") -> None:
        name = str(name) + "-" + which
        assert name not in self.__class__.stat
        self.name = name
        self.which = which
        self.stat = Stat()
        self.__class__.stat[name] = self.stat

    def __call__(self, m, i, o):
        if self.which == "output":
            value = o
        elif self.which == "input":
            value = i

        assert isinstance(value, torch.Tensor)
        self.stat.update(value.squeeze())


if __name__ == "__main__":
    stat = Stat()
    stat.update(torch.tensor([[1.0, 42, 53], [3.0, 32, 55]]))
    stat.update(torch.tensor([[6.0, 32, 35]]))
    stat.update(torch.tensor([[5.0, 72, 57]]))
    print(stat)
