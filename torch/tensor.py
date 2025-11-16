# ###
# Tensor Autograd
# calculate Grad and assign Backward function, Ref to diff and chain rule. grad += diff of target * diff of global
# calculate Layer normalization
# h = f(g/sigma*(a - mu) + b)
# mu = 1/H * sum of a
# sigma = sqrt(1/H * sum of (a - mu)**2)
# ###

import numpy as np

def is_tensor(func):
    def wrapper(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return func(self, other)
    return wrapper

class Tensor():
    '''A simple Tensor class for automatic differentiation.'''
    def __init__(self, value : np.ndarray, children=()):
        self.value = value
        self.grad = np.zeros_like(value)
        self._prev = set(children)
        self._backward = lambda: None

    def backward(self):
        '''a backward chainrule collecting the gradient of actions were made.'''
        topo = []
        visited = set()
        #dfs
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        self.grad = np.ones_like(self.value)

        for t in reversed(topo):
            t._backward()
    
    def zero_grad(self):
        '''Reset the gradient to zero.'''
        self.grad = np.zeros_like(self.value)
    
    @is_tensor
    def __add__(self, other):
        out = Tensor(self.value + other.value, (self, other))

        def _backward():
            grad_self = out.grad
            while len(grad_self.shape) > len(self.value.shape):
                grad_self = grad_self.sum(axis=0)
            for i, dim in enumerate(self.value.shape):
                if dim == 1:
                    grad_self = grad_self.sum(axis=i, keepdims=True)
            self.grad += grad_self

            grad_other = out.grad
            while len(grad_other.shape) > len(other.value.shape):
                grad_other = grad_other.sum(axis=0)
            for i, dim in enumerate(other.value.shape):
                if dim == 1:
                    grad_other = grad_other.sum(axis=i, keepdims=True)
            other.grad += grad_other
            
        out._backward = _backward

        return out
    
    @is_tensor
    def __mul__(self, other):
        out = Tensor(self.value * other.value, (self, other))

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward

        return out
    
    @is_tensor
    def __sub__(self, other):
        out = Tensor(self.value - other.value, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += -1 * out.grad
        out._backward = _backward
        return out
    
    @is_tensor
    def __truediv__(self, other):
        out = Tensor(self.value / other.value, (self, other))
        def _backward():

            grad_self = (1 / other.value) * out.grad
            while len(grad_self.shape) > len(self.value.shape):
                grad_self = grad_self.sum(axis=0)
            for i, dim in enumerate(self.value.shape):
                if dim == 1:
                    grad_self = grad_self.sum(axis=i, keepdims=True)
            self.grad += grad_self

            grad_other = (-self.value / (other.value **2)) * out.grad
            while len(grad_other.shape) > len(other.value.shape):
                grad_other = grad_other.sum(axis=0)
            for i, dim in enumerate(other.value.shape):
                if dim == 1:
                    grad_other = grad_other.sum(axis=i, keepdims=True)
            other.grad += grad_other
        out._backward = _backward
        return out
    
    @is_tensor
    def __matmul__(self, other):
        out = Tensor(self.value @ other.value, (self, other))

        def _backward():
            # B, L, K = self.value.shape
            # M = out.grad.shape[-1]
            # X_reshaped = self.value.reshape(B * L, K) # (80, 128)
            # dY_reshaped = out.grad.reshape(B * L, M) # (80, 512)
            # dW = X_reshaped.T @ dY_reshaped # (128, 80) @ (80, 512) -> (128, 512)
            # grad_for_other = dW
            if self.value.ndim > 2 and other.value.ndim == 2:
                grad_self = np.einsum('blm, km -> blk', out.grad, other.value)
                grad_other = np.einsum('blk, blm -> km', self.value, out.grad)
            elif self.value.ndim == other.value.ndim and self.value.ndim > 2:
                grad_self = np.einsum('bhlm, bhkm -> bhlk', out.grad, other.value)
                grad_other = np.einsum('bhlk, bhlm -> bhkm', self.value, out.grad)
            else:
                raise NotImplementedError(f"Matmul backward for ndim {self.value.ndim} @ {other.value.ndim} is not implemented.")
            self.grad += grad_self
            other.grad += grad_other
        out._backward = _backward

        return out
    
    def __neg__(self):
        out = Tensor(-self.value, (self,))
        def _backward():
            self.grad += -1 * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supporting int/float powers"
        out = Tensor(self.value ** power, (self,))

        def _backward():
            self.grad += (power * self.value**(power -1)) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        out = Tensor(np.exp(self.value), (self,))

        def _backward():
            self.grad += out.value * out.grad
        out._backward = _backward

        return out
    def log(self):
        out = Tensor(np.log(self.value), (self,))

        def _backward():
            self.grad += (1 / self.value) * out.grad
        out._backward = _backward

        return out
    
    def softmax(self):
        max_val = np.max(self.value, axis=-1, keepdims=True)
        stable_val = self.value - max_val
        exp_value = np.exp(stable_val)
        sum_exp =np.sum(exp_value, axis=-1, keepdims=True)
        out_val = exp_value /(sum_exp + 1e-9)
        out = Tensor(out_val, (self,))
        
        def _backward():
            self.grad += (out.value * out.grad) - out.value * np.sum(out.value * out.grad, axis=-1, keepdims=True)
        out._backward = _backward

        return out
    
    def sqrt(self):
        out = Tensor(np.sqrt(self.value), (self,))

        def _backward():
            self.grad += (0.5 / np.sqrt(self.value)) * out.grad
        out._backward = _backward
    
    def transpose(self, axes):
        out = Tensor(np.transpose(self.value, axes), (self,))
        
        def _backward():
            self.grad += out.grad.transpose(axes)
        out._backward = _backward
        return out
    
    def reshape(self, shape):
        out = Tensor(self.value.reshape(shape), (self,))

        def _backward():
            self.grad += out.grad.reshape(self.value.shape)
        out._backward = _backward

        return out
    
    def relu(self):
        out = Tensor(np.maximum(self.value, 0), (self,))

        def _backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = _backward

        return out
    
    def sin(self):
        out = Tensor(np.sin(self.value), (self,))
        def _backward():
            self.grad += np.cos(self.value) * out.grad
        out._backward = _backward
        return out
    
    def cos(self):
        out = Tensor(np.cos(self.value), (self,))
        def _backward():
            self.grad += -1 * np.sin(self.value) * out.grad
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Tensor(value={self.value}, grad={self.grad})"

    def layerNorm(self, g : Tensor, b: Tensor, eps=1e-5):
        a = self.value
        H = a.shape[-1]
        mu = np.mean(a, axis=-1, keepdims=True)
        var = np.var(a, axis=-1, keepdims=True)
        sigma = np.sqrt(var + eps)
        a_norm = (a - mu) / sigma
        normal = (g.value * a_norm) + b.value
        out = Tensor(normal, (self, g, b))
        def _backward():
            d_normal = out.grad
            sum_axis = tuple(range(len(a.shape) - 1))
            d_g = np.sum(d_normal * a_norm, axis=sum_axis, keepdims=True)
            d_b = np.sum(d_normal, axis=sum_axis, keepdims=True)
            g.grad += d_g
            b.grad += d_b
            d_anorm = d_normal * g.value
            d_var = np.sum(d_anorm * (a - mu) * (-0.5) * np.power(var + eps, -1.5), axis=-1, keepdims=True)
            d_mu = np.sum(d_anorm * (-1/sigma), axis=-1, keepdims=True) + d_var * np.sum(-2 * (a - mu), axis=-1, keepdims=True) / H
            da = (d_anorm / sigma) + (d_var * 2 * (a - mu) / H) + (d_mu / H)
            self.grad += da
        out._backward = _backward
        return out
    def cross_entropy(self, Y_true):
        epsilon = 1e-9
        loss_val = -np.sum(Y_true.value * np.log(self.value + epsilon))
        out = Tensor(np.array(loss_val), (self, Y_true)) 

        def _backward():
            self.grad += -Y_true.value / (self.value + epsilon)
            
        out._backward = _backward
        return out