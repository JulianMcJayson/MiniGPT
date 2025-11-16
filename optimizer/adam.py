# ###
# Adam adaptive moment estimation
# parameter: alpha, t, b1,b2, m, v, theta, eps
# while theta not converged
# update time
# t += 1
# update bias
# m(t) = b1 * m(t - 1) + (1 - b1) * g
# v(t) = b2 * v(t - 1) + (1 - b2) * g**2
# bias correction
# m'(t) = m(t-1)/(1 - b1**t)
# v'(t) = v(t-1)/(1 - b2**t)
# update theta
# theta(t) = theta(t-1) - alpha * (m'(t)) / sqrt(v'(t) + eps)
# return theta
# ###

from torch.tensor import Tensor
import numpy as np

class Adam():
    def __init__(self, param : list[Tensor], alpha = 1e-3, b1 = 0.9, b2 = 0.999, eps = 10**-8) -> None:
        self.theta = param
        self.alpha = alpha
        self.b1 = b1
        self.b2 = b2
        self.t = 0
        self.eps = eps
        self.m = [np.zeros_like(p.value) for p in param]
        self.v = [np.zeros_like(p.value) for p in param]
    def step(self):
        self.t += 1
        for i, p in enumerate(self.theta):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1-self.b2) * p.grad**2
            m_hat = self.m[i]/(1 - self.b1**self.t)
            v_hat = self.v[i]/(1 - self.b2**self.t)
            p.value -= self.alpha * m_hat / np.sqrt(v_hat + self.eps)
    def zero_grad(self):
        for p in self.theta:
            p.zero_grad()
            