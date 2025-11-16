# ###
# Training Model
# init model
# get params from model
# init optimizer with params
# fake data (optional)
# training epoch
# zero grad at optimizer
# predicts
# softmax model on top
# create loss with answer
# backward the loss
# step the optimizer
# ###

from transformer.model import MiniGPT
from torch.tensor import Tensor
from optimizer.adam import Adam
import numpy as np

model = MiniGPT(dmodel=128, numheads=8, dk=64, dv=64, dff=512)
params = model.get_param()
optimizer = Adam(params)
B =4
L = 20
D = 128
X = Tensor(np.random.rand(B,L,D))
Y_true_np = np.zeros((B, L, D))
Y_true_np[:, :, 10] = 1.0
Y_true = Tensor(Y_true_np) 
epoch = 10

for e in range(epoch):
    optimizer.zero_grad()

    result = model(X)

    result_sm = result.softmax()

    loss = result_sm.cross_entropy(Y_true)

    print(f"Epoch {e + 1}: loss={loss.value:.4f}")

    loss.backward()

    optimizer.step()