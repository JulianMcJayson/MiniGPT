# MiniGPT - Transformer from Scratch in Pure NumPy

A fully-functional, decoder-only GPT-like transformer built entirely from scratch using only NumPy. Includes custom autograd engine, optimizer, and complete training pipeline.

## Overview

This project implements a character-level language model inspired by GPT architecture, written from the ground up without PyTorch/TensorFlow. Every component—from tensor operations to backpropagation to the transformer itself—is hand-coded.

**Key Achievement**: Demonstrates deep understanding of transformer internals, automatic differentiation, and optimization algorithms by implementing everything manually.

## Project Structure

```
├── torch/
│   └── tensor.py              # Custom autograd tensor with backward()
├── optimizer/
│   └── adam.py                # Adam optimizer implementation
├── transformer/
│   └── model.py               # MiniGPT decoder-only architecture
├── example.py                 # Training script with text generation
└── hamlet.txt                 # Training corpus
```

## Architecture Details

### Custom Tensor (`torch/tensor.py`)
Implements automatic differentiation with computational graph:
- **Backward pass**: DFS topological sort for gradient propagation
- **Operations**: `+`, `-`, `*`, `/`, `@`, `exp`, `log`, `sqrt`, `relu`, `sin`, `cos`
- **Advanced ops**: `softmax`, `layerNorm`, `cross_entropy`, `lookup` (embedding)
- **Broadcasting**: Full NumPy-style broadcasting support with gradient reduction

### Adam Optimizer (`optimizer/adam.py`)
Textbook implementation of adaptive moment estimation:
```
m_t = β₁·m_{t-1} + (1-β₁)·g_t
v_t = β₂·v_{t-1} + (1-β₂)·g_t²
m̂_t = m_t/(1-β₁ᵗ)
v̂_t = v_t/(1-β₂ᵗ)
θ_t = θ_{t-1} - α·m̂_t/√(v̂_t + ε)
```
Default: `α=1e-3`, `β₁=0.9`, `β₂=0.999`, `ε=1e-8`

### MiniGPT Model (`transformer/model.py`)
Decoder-only transformer with:
- **Multi-head self-attention** with causal masking
- **Sinusoidal positional encoding**: PE(pos, 2i) = sin(pos/10000^(2i/d_model))
- **Layer normalization** (pre-norm configuration)
- **Feed-forward network** with ReLU activation
- **Residual connections** around attention and FFN blocks

**Parameters**:
- Vocabulary size: Dynamic (based on unique characters)
- d_model: 128 (embedding dimension)
- Heads: 8
- d_k, d_v: 64 (query/key and value dimensions)
- d_ff: 512 (FFN hidden size)

## Training Pipeline

### Data Preprocessing
```python
# Clean text (alphanumeric only)
clean_data = re.sub(r'[^a-zA-Z0-9\s]', '', raw_text)

# Character-level tokenization
words = sorted(set(clean_data))
string_to_index = {s:i for i,s in enumerate(words)}
dataset = [string_to_index[w] for w in clean_data]
```

### Training Loop
```python
for epoch in range(1000):
    optimizer.zero_grad()
    
    # Get batch (B=8, L=16)
    x, y = get_batch(dataset, batch_size=8, block_size=16)
    
    # Forward pass
    logits = model(x)
    probs = logits.softmax()
    loss = probs.cross_entropy(y)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
```

### Text Generation
Autoregressive sampling with temperature-based multinomial selection:
```python
def generate(model, start_text, max_new_tokens=100):
    # Convert start text to indices
    # Loop: predict next token, sample from distribution, append
    # Return decoded string
```

## Installation

```bash
# No dependencies except NumPy
pip install numpy
```

## Usage

### Train the Model
```bash
python example.py
```

Trains on `hamlet.txt` for 1000 epochs, prints loss every 100 steps, then generates 100 characters starting with "The ".

### Customize Hyperparameters
```python
model = MiniGPT(
    vocab_size=vocab_size,
    dmodel=128,        # Increase for larger capacity
    numheads=8,        # More heads = more parallel attention
    dk=64,             # Query/key dimension per head
    dv=64,             # Value dimension per head
    dff=512            # FFN hidden layer size
)

optimizer = Adam(params, alpha=1e-3)  # Learning rate
```

## Implementation Highlights

### 1. Causal Masking
```python
mask = np.triu(np.ones((L, L)), k=1) * -1e9
scores.value += mask  # Prevents attending to future tokens
```

### 2. Layer Normalization Backward Pass
Full analytical gradient computation for:
- Normalized values
- Variance term
- Mean term
- Scale (γ) and bias (β) parameters

### 3. Einsum-based Matmul Gradients
```python
# For 3D @ 2D: einsum('blm,km->blk')
# For 4D @ 4D: einsum('bhlm,bhkm->bhlk')
```

### 4. Embedding Lookup with Gradient Accumulation
```python
np.add.at(grad_we, indices.value, out.grad)
# Correctly handles repeated indices
```

## Performance Characteristics

**Typical Training**:
- Dataset: ~50K characters (Hamlet)
- Epochs: 1000
- Time: ~10-30 minutes (CPU-only)
- Final loss: ~2.5-3.5 (character-level cross-entropy)

**Generated Text Quality**: Learns character patterns, word structures, and basic grammar. Output is coherent at character level but may lack long-range semantic consistency (expected for small model/dataset).

## Key Differences from Production Code

| Aspect | This Implementation | PyTorch/Production |
|--------|---------------------|-------------------|
| Tensor ops | Pure NumPy | Optimized C++/CUDA |
| Autograd | Manual graph construction | Dynamic computation graph |
| Memory | Full history stored | Gradient checkpointing |
| Speed | ~100x slower | GPU-accelerated |
| Purpose | Educational | Production-ready |

## Reference Papers

1. Attention Is All You Need
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017)

2. Adam: A Method for Stochastic Optimization
Kingma, D. P., & Ba, J. (2014)
Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

3. Layer Normalization
Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016)
Layer normalization. arXiv preprint arXiv:1607.06450.

4. Neural Machine Translation by Jointly Learning to Align and Translate
Bahdanau, D., Cho, K., & Bengio, Y. (2014)
Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
