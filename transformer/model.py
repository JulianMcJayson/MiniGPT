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
    def __init__(self,vocab_size :int, dmodel : int, numheads : int, dk : int, dv : int, dff : int) -> None:
        self.numheads = numheads
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.wE = Tensor(np.random.rand(vocab_size, dmodel) * 0.02)
        self.w1 = Tensor(np.random.rand(self.dmodel, dff) * 0.02)
        self.w2 = Tensor(np.random.rand(dff, self.dmodel) * 0.02)
        self.b1 = Tensor(np.random.rand(1, dff) * 0.02)
        self.b2 = Tensor(np.random.rand(1, self.dmodel) * 0.02)
        self.wQ = Tensor(np.random.rand(dmodel,  numheads * dk) * 0.02)
        self.wK = Tensor(np.random.rand(dmodel,  numheads * dk) * 0.02)
        self.wV = Tensor(np.random.rand(dmodel,  numheads * dv) * 0.02)
        self.wO = Tensor(np.random.rand(numheads * dv, dmodel) * 0.02)
        self.gainlr1 = Tensor(np.ones((1, 1, dmodel)) * 0.02)
        self.biaslr1 = Tensor(np.zeros((1, 1, dmodel)) * 0.02)
        self.gainlr2 = Tensor(np.ones((1, 1, dmodel)) * 0.02)
        self.biaslr2 = Tensor(np.zeros((1, 1, dmodel)) * 0.02)
        self.w_final = Tensor(np.random.rand(dmodel, vocab_size) * 0.02)
        self.dk = dk
        self.dv = dv
        self.dq = dmodel // numheads
    
    def get_param(self):
        return [self.wE, self.w1, self.w2, self.b1, self.b2, self.wQ, self.wK, self.wV, self.wO, self.gainlr1, self.biaslr1, self.gainlr2, self.biaslr2, self.w_final]
    
    def _split_head(self, X : Tensor):
        B, L, _ = X.value.shape
        H = self.numheads
        X_reshaped = X.reshape((B, L, H, self.dk))
        return X_reshaped.transpose((0, 2, 1, 3))
    
    def _reshape_head(self, X : Tensor):
        B, H, L, _ = X.value.shape
        X_reshaped = X.transpose((0, 2, 1, 3))
        return X_reshaped.reshape((B, L, H * self.dv))

    def __call__(self, indices : Tensor):
        
        x_emb = self.wE.lookup(indices)
        x_pos = self.PE(x_emb)
        X_c = x_emb + x_pos

        Q = X_c @ self.wQ
        K = X_c @ self.wK
        V = X_c @ self.wV
        Q_split = self._split_head(Q)
        K_split = self._split_head(K)
        V_split = self._split_head(V)

        scores = Q_split @ K_split.transpose((0, 1, 3, 2)) / np.sqrt(self.dk)

        B, H, L, _ = scores.value.shape
        mask = np.triu(np.ones((L, L)), k=1) * -1e9
        scores.value += mask

        heads = scores.softmax() @ V_split
        
        attention_output =  self._reshape_head(heads) @ self.wO
        attention_output_R = attention_output + X_c
        
        norm_layer = attention_output_R.layerNorm(self.gainlr1, self.biaslr1)
        ffn_out = self.FFN(norm_layer)
        ffn_out_R = ffn_out + norm_layer
        
        features = ffn_out_R.layerNorm(self.gainlr2, self.biaslr2)
        
        logits = features @ self.w_final 
        
        return logits
    
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
