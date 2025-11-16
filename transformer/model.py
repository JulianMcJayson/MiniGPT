# ###
# Transformers
# Attention = (Q * K^T)/sqrt(dk) * V
# Head(n) = Attention(Q*Wi(Q), K*Wi(K), V*Wi(V))
# Multi-Head = Concat(Head(1)...Head(n))*Wi(O)
# FFN = max(0, wx + b)w + b
# Layer Normalization
# Position Encoding
# PE(pos, 2i) = sin(pos/10000**2i/dmodel)
# PE(pos, 2i + 1) = cos(pos/10000**2i/dmodel)
# Layer Normalization
# Output
# ###

import numpy as np
from torch.tensor import Tensor
class MiniGPT():
    """A simplified implementation of a GPT-like transformer model.
    Decoding-Only Transformer
    """
    def __init__(self, dmodel : int, numheads : int, dk : int, dv : int, dff : int) -> None:
        self.numheads = numheads
        self.dmodel = dmodel
        self.w1 = Tensor(np.random.rand(self.dmodel, dff))
        self.w2 = Tensor(np.random.rand(dff, self.dmodel))
        self.b1 = Tensor(np.random.rand(1, dff))
        self.b2 = Tensor(np.random.rand(1, self.dmodel))
        self.wQ = Tensor(np.random.rand(dmodel,  numheads * dk))
        self.wK = Tensor(np.random.rand(dmodel,  numheads * dk))
        self.wV = Tensor(np.random.rand(dmodel,  numheads * dv))
        self.wO = Tensor(np.random.rand(numheads * dv, dmodel))
        self.dk = dk
        self.dv = dv
        self.dq = dmodel // numheads
    
    def _split_head(self, X : Tensor):
        B, L, _ = X.value.shape
        H = self.numheads
        X_reshaped = X.reshape((B, L, H, self.dk))
        return X_reshaped.transpose((0, 2, 1, 3))
    
    def _reshape_head(self, X : Tensor):
        B, H, L, _ = X.value.shape
        X_reshaped = X.transpose((0, 2, 1, 3))
        return X_reshaped.reshape((B, L, H * self.dv))

    def __call__(self, X : Tensor):
        X_c = X + self.PE(X)

        Q = X_c @ self.wQ
        K = X_c @ self.wK
        V = X_c @ self.wV
        Q_split = self._split_head(Q)
        K_split = self._split_head(K)
        V_split = self._split_head(V)

        heads : Tensor = (Q_split @ K_split.transpose((0, 1, 3, 2)) / np.sqrt(self.dk)).softmax() @ V_split
        attention_output =  self._reshape_head(heads) @ self.wO
        attention_output_R = attention_output + X_c
        norm_layer = attention_output_R.layerNorm()
        ffn_out = self.FFN(norm_layer)
        ffn_out_R = ffn_out + norm_layer
        return ffn_out_R.layerNorm()
    
    def FFN(self, X : Tensor):
        relu = ((X @ self.w1) + self.b1).relu()
        return (relu @ self.w2) + self.b2
    
    def _get_denominator(self):
        """Calculates the denominator term for positional encoding."""
        # Shape: (d_model,)
        indices = np.arange(self.dmodel)
        # Shape: (d_model,) -> e.g., [0, 0, 2, 2, 4, 4, ...] for d_model=8
        exponent = 2 * (indices // 2) / self.dmodel
        # Shape: (1, d_model) for broadcasting
        return (10000 ** exponent)[np.newaxis, :]
        
    def PE(self, X : Tensor):
        """Generates the Positional Encoding matrix."""
        _, L, _ = X.value.shape
        denominator = self._get_denominator()
        # Shape: (L, 1)
        pos = np.arange(L)[:, np.newaxis]
        # Broadcasting (L, 1) / (1, d_model) -> (L, d_model)
        arg_matrix = pos / denominator
        pe = np.zeros((L, self.dmodel))
        pe[:, 0::2] = np.sin(arg_matrix[:, 0::2])
        pe[:, 1::2] = np.cos(arg_matrix[:, 1::2])
        return Tensor(pe)
