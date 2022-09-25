import torch
import torch.optim.optimizer
from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.optim.mixin import OptimMixin
from geoopt.manifolds import Scaled

__all__ = ["RiemannianSGD"]


class RiemannianSGD(OptimMixin, torch.optim.Optimizer):
    r"""
    Riemannian Stochastic Gradient Descent with the same API as :class:`torch.optim.SGD`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float
        learning rate
    momentum : float (optional)
        momentum factor (default: 0)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    dampening : float (optional)
        dampening for momentum (default: 0)
    nesterov : bool (optional)
        enables Nesterov momentum (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        stabilize=None,
        manifold=None
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            manifold=manifold
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults, stabilize=stabilize)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                learning_rate = group["lr"]
                manifold = group['manifold']
                group["step"] += 1
                point = []
                grads = []
                for p in group["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianSGD does not support sparse gradients, use SparseRiemannianSGD instead"
                        )
                    point.append(p.data.view(1, -1))
                    grads.append(grad.view(1, -1))
                point = torch.cat(point, dim=1).squeeze()
                grads = torch.cat(grads, dim=1).squeeze()

                state = self.state[id(group["params"])]
                # State initialization
                if len(state) == 0:
                    if momentum > 0:
                        state["momentum_buffer"] = grads.clone()

                grads.add_(point, alpha=weight_decay)

                # grads = manifold.egrad2rgrad(point, grads)
                grads = grads - (point * grads).sum() * point / point.pow(2).sum()
                if momentum > 0:
                    momentum_buffer = state["momentum_buffer"]
                    momentum_buffer.mul_(momentum).add_(grads, alpha=1 - dampening)
                    if nesterov:
                        grads = grads.add_(momentum_buffer, alpha=momentum)
                    else:
                        grads = momentum_buffer
                    # we have all the things projected
                    # new_point, new_momentum_buffer = manifold.retr_transp(
                    #     point, -learning_rate * grads, momentum_buffer
                    # )
                    tmp = point - learning_rate * grads
                    if isinstance(manifold, Scaled):
                        new_point = tmp * manifold.scale / torch.norm(tmp)
                        new_momentum_buffer = momentum_buffer - (new_point * momentum_buffer).sum() * new_point / manifold.scale.pow(2)
                    else:
                        new_point = tmp / torch.norm(tmp)
                        new_momentum_buffer = momentum_buffer - (new_point * momentum_buffer).sum() * new_point

                    momentum_buffer.copy_(new_momentum_buffer)
                    # use copy only for user facing point
                    point.copy_(new_point)
                else:
                    new_point = manifold.retr(point, -learning_rate * grads)
                    # t = torch.norm(new_point, p=2)
                    # t = torch.norm(grads)
                    point.copy_(new_point)

                if (
                    group["stabilize"] is not None
                    and group["step"] % group["stabilize"] == 0
                ):
                    self.stabilize_group(group)

                start = 0
                stop = 0
                for p in group["params"]:
                    stop += p.numel()
                    new_p = point[start: stop].reshape(p.shape)
                    p.data = new_p
                    start = stop
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            manifold = p.manifold
            momentum = group["momentum"]
            p.copy_(manifold.projx(p))
            if momentum > 0:
                param_state = self.state[p]
                if not param_state:  # due to None grads
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.copy_(manifold.proju(p, buf))
