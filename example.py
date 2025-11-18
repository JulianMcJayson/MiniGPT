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
import re

#dataset
with open("hamlet.txt", "+r") as file:
    data = file.read()

clean_data : str = re.sub(r'[^a-zA-Z0-9\s]', '', data)

words = sorted(list(set(clean_data)))
string_to_index = {s:i for i, s in enumerate(words)}
index_to_string = {i:s for i, s in enumerate(words)}
vocab_size = len(index_to_string)
print(vocab_size)
print(len(words))
print(len(string_to_index))

dataset = [string_to_index[w] for w in clean_data]

def get_batch(dataset, block_size, batch_size, vocab_size):
    idx = np.random.randint(0, len(dataset) - block_size, (batch_size,))

    x_stack = []
    y_stack = []

    for i in idx:
        x_cluster = dataset[i: i + block_size]
        x_stack.append(x_cluster)
        y_cluster = dataset[i + 1: i + block_size + 1]
        y_stack.append(y_cluster)
    
    inputs = np.array(x_stack)
    targets = np.array(y_stack)

    # one_hot = np.zeros((batch_size, block_size, vocab_size))
    # for ba in range(batch_size):
    #     for bl in range(block_size):
    #         idx = targets[ba, bl]
    #         one_hot[ba, bl,idx] = 1.0
    
    batch_x = Tensor(inputs)
    batch_y = Tensor(targets)
    return batch_x, batch_y

model = MiniGPT(vocab_size=vocab_size, dmodel=128, numheads=8, dk=64, dv=64, dff=512)
params = model.get_param()
optimizer = Adam(params, alpha=1e-3)
B = 8
L = 16

epoch = 1000

for e in range(epoch):
    optimizer.zero_grad()

    x, y = get_batch(dataset, B, L, vocab_size)

    result = model(x)

    result_sm = result.softmax()

    loss = result_sm.cross_entropy(y)

    if e % 100 == 0:
        print(f"Epoch {e}: loss={loss.value:.4f}")

    loss.backward()

    optimizer.step()


def generate(model, start_text, max_new_tokens, block_size):
    idx = [string_to_index[c] for c in start_text]
    idx = np.array(idx).reshape(1, -1)
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        
        input_tensor = Tensor(idx_cond)
        logits = model(input_tensor)

        last_logits = logits.value[:, -1, :]
        probs = np.exp(last_logits) / np.sum(np.exp(last_logits), axis=-1, keepdims=True)
        next_index = np.random.choice(len(probs[0]), p=probs[0])
        
        idx = np.concatenate((idx, [[next_index]]), axis=1)

    output_text = "".join([index_to_string[i] for i in idx[0]])
    return output_text

print("-" * 50)
print("output:")

start_str = "The " 
generated_text = generate(model, start_str, max_new_tokens=100, block_size=L)

print(generated_text)
print("-" * 50)