# Neural Network Architecture Documentation

## Overview

This document provides detailed technical specifications for the neural network architecture used in the Quantum Neural ODE (QNODE) project. The model implements a Latent Neural Ordinary Differential Equation framework for learning quantum dynamics.

## Architecture Components

### 1. LatentODEfunc - Core Dynamics Network

**Purpose**: Learns the continuous-time dynamics in the latent space

**Architecture**:
```python
class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=6, nhidden=53):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)      # 6 → 53
        self.fc2 = nn.Linear(nhidden, nhidden)         # 53 → 53
        self.fc3 = nn.Linear(nhidden, nhidden)         # 53 → 53
        self.fc4 = nn.Linear(nhidden, latent_dim)      # 53 → 6
```

**Layer Details**:
- **Input dimension**: 6 (latent state variables)
- **Hidden layers**: 2 layers with 53 neurons each
- **Output dimension**: 6 (latent state derivatives)
- **Activation function**: ELU (Exponential Linear Unit)
- **Total parameters**: ~6,000 parameters

**Forward Pass**:
```python
def forward(self, t, x):
    out = self.fc1(x)
    out = self.elu(out)
    out = self.fc2(out)
    out = self.elu(out)
    out = self.fc3(out)
    out = self.elu(out)
    out = self.fc4(out)
    return out
```

### 2. RecognitionRNN - Encoder Network

**Purpose**: Encodes observation sequences into latent initial conditions

**Architecture**:
```python
class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=6, obs_dim=3, nhidden=53, nbatch=1080):
        super(RecognitionRNN, self).__init__()
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)    # (3+53) → 53
        self.h2o = nn.Linear(nhidden, latent_dim * 2)       # 53 → 12
```

**Layer Details**:
- **Input dimension**: 3 (observation variables) + 53 (hidden state) = 56
- **Hidden dimension**: 53 neurons
- **Output dimension**: 12 (6 for mean + 6 for log variance)
- **Activation function**: Tanh
- **Total parameters**: ~3,500 parameters

**Forward Pass**:
```python
def forward(self, x, h):
    combined = torch.cat((x, h), dim=1)
    h = torch.tanh(self.i2h(combined))
    out = self.h2o(h)
    return out, h
```

### 3. Decoder Network

**Purpose**: Decodes latent states back to observable quantum states

**Architecture**:
```python
class Decoder(nn.Module):
    def __init__(self, latent_dim=6, obs_dim=3, nhidden=53, extra=True):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, nhidden)      # 6 → 53
        self.fc2 = nn.Linear(nhidden, nhidden)         # 53 → 53
        self.fc3 = nn.Linear(nhidden, obs_dim)         # 53 → 3
```

**Layer Details**:
- **Input dimension**: 6 (latent variables)
- **Hidden layer**: 53 neurons
- **Output dimension**: 3 (observable quantum state parameters)
- **Activation function**: Tanh
- **Total parameters**: ~3,500 parameters

**Forward Pass**:
```python
def forward(self, z):
    out = self.fc1(z)
    out = self.tanh(out)
    out = self.fc2(out)
    out = self.tanh(out)
    out = self.fc3(out)
    return out
```

## Model Integration

### Complete Architecture

The three components are integrated in the `latent_ode` class:

```python
class latent_ode:
    def __init__(self, obs_dim=3, latent_dim=6, nhidden=53, 
                 rnn_nhidden=53, lr=0.007, batch=1080, beta=1):
        self.func = LatentODEfunc(latent_dim, nhidden)
        self.rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch)
        self.dec = Decoder(latent_dim, obs_dim, nhidden, extra=True)
```

### Parameter Count Summary

| Component | Parameters | Description |
|-----------|------------|-------------|
| LatentODEfunc | ~6,000 | Core dynamics learning |
| RecognitionRNN | ~3,500 | Sequence encoding |
| Decoder | ~3,500 | Latent-to-observable mapping |
| **Total** | **~13,000** | **Complete model** |

## Training Configuration

### Hyperparameters
- **Learning rate**: 0.007
- **Batch size**: 1080
- **Beta (KL weight)**: 1.0
- **Optimizer**: Adam
- **Latent dimension**: 6
- **Hidden dimension**: 53

### Loss Function
```python
Loss = -log p(x|z) + β × KL(q(z|x) || p(z))
```

Where:
- `p(x|z)` is the reconstruction likelihood
- `KL(q(z|x) || p(z))` is the KL divergence regularization
- `β` controls the regularization strength

## Activation Functions

### ELU (Exponential Linear Unit)
Used in the LatentODEfunc:
```python
ELU(x) = x if x > 0
         α(e^x - 1) if x ≤ 0
```
- Provides smooth derivatives
- Reduces vanishing gradient problems
- Maintains mean activation closer to zero

### Tanh (Hyperbolic Tangent)
Used in RecognitionRNN and Decoder:
```python
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- Bounded output [-1, 1]
- Zero-centered
- Suitable for RNN hidden states

## Implementation Details

### ODE Integration
The model uses `torchdiffeq` for solving the neural ODE:
```python
pred_z = odeint(self.func, z0, ts)
```
- Adaptive step-size solver
- Automatic differentiation through the solver
- Handles stiff differential equations

### Variational Inference
The encoder outputs mean and log-variance:
```python
qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
epsilon = torch.randn(qz0_mean.size())
z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
```

### Reconstruction Process
1. **Encoding**: Sequence → latent initial condition
2. **Integration**: Solve ODE in latent space
3. **Decoding**: Latent trajectory → observable trajectory

## Model Variants

### Configuration Options
- **obs_dim**: Observable dimension (3 for quantum states)
- **latent_dim**: Latent space dimension (6 for current model)
- **nhidden**: Hidden layer size (53 for current model)
- **extra_decode**: Additional decoder layer (enabled)

### Pre-trained Models
- **open_6_53_53_0.007**: Amplitude damping channel
- **closed_6_48_48_0.004**: Closed quantum system
- **two_8_170_170_0.002**: Two-level system

## Performance Characteristics

### Computational Complexity
- **Training**: O(batch_size × sequence_length × hidden_dim²)
- **Inference**: O(sequence_length × hidden_dim²)
- **Memory**: ~50MB for model parameters

### Convergence Properties
- **Training epochs**: 1000-5000 for convergence
- **Gradient flow**: Stable with proper regularization
- **Overfitting**: Controlled by KL divergence term

## Future Enhancements

1. **Attention mechanisms** for better sequence modeling
2. **Residual connections** for deeper networks
3. **Batch normalization** for training stability
4. **Dropout layers** for regularization
5. **Graph neural networks** for multi-particle systems

---

*This architecture represents a novel application of neural ODEs to quantum dynamics, combining the strengths of continuous-time modeling with variational inference for quantum state prediction.*
