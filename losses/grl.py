# utils/grl.py
import torch


class GradientReverseLayer(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GRL(torch.nn.Module):
    def __init__(self):
        super(GRL, self).__init__()
        self.grl = GradientReverseLayer.apply
        self._enabled = False

    def forward(self, x):
        if self._enabled:
            return self.grl(x)
        return x

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False