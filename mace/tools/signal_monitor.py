import torch
class SignalTap:
    """
    Registers forward & backward hooks and stores mean |activation| / grad-norm
    even when the module returns (or receives) tuples.
    """
    def __init__(self, modules, every=25):
        self.every  = every
        self.i_step = 0
        self.stats  = {m: {"act": [], "grad": []} for m in modules}

        def tensor_mean(x):  # recursively handle tuple / list
            if torch.is_tensor(x):
                return x.detach().abs().mean()
            if isinstance(x, (tuple, list)):
                ts = [tensor_mean(t) for t in x if isinstance(t, (torch.Tensor, tuple, list))]
                return torch.stack(ts).mean() if ts else torch.tensor(0.)
            return torch.tensor(0.)

        for m in modules:
            m.register_forward_hook(
                lambda mod, inp, out, m=m: self.stats[m]["act"].append(tensor_mean(out))
            )
            m.register_full_backward_hook(
                lambda mod, gin, gout, m=m: (
                    self.stats[m]["grad"].append(
                        torch.stack([g.detach().norm()
                                     for g in gout if torch.is_tensor(g)]).mean()
                            if gout else torch.tensor(0.))
                )
            )

    def step(self):
        self.i_step += 1
        if self.i_step % self.every == 0:
            print(f"\n── diagnostics @ step {self.i_step}")
            for m, d in self.stats.items():
                a = torch.stack(d["act"]).mean().item()  if d["act"]  else 0.
                g = torch.stack(d["grad"]).mean().item() if d["grad"] else 0.
                print(f"{m.__class__.__name__:<35}  |act|={a:8.3e}   |grad|={g:8.3e}")
                d["act"].clear(); d["grad"].clear()
