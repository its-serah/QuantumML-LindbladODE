"""
Pauli Expectation Values Analysis for Amplitude Damping - Plus State
====================================================================

This script compares Neural ODE predictions with theoretical Lindblad evolution
for the amplitude damping channel, starting from the |+⟩ state.

The plus state is defined as |+⟩ = (|0⟩ + |1⟩)/√2, which is an eigenstate of σₓ.
This provides a different perspective on the amplitude damping dynamics compared
to the computational basis states.

Author: Quantum ML Research Team
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torchdiffeq import odeint
import os

# Set matplotlib parameters for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class LatentODEfunc(nn.Module):
    """Neural ODE function for learning quantum dynamics in latent space"""
    def __init__(self, latent_dim, nhidden):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, nhidden)
        self.fc4 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        out = self.elu(out)
        out = self.fc4(out)
        return out

class RecognitionRNN(nn.Module):
    """RNN encoder for variational inference"""
    def __init__(self, latent_dim, obs_dim, nhidden, nbatch):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)

class Decoder(nn.Module):
    """Decoder network mapping latent states to observables"""
    def __init__(self, latent_dim, obs_dim, nhidden):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = torch.tanh(out)
        out = self.fc3(out)
        return out

def amplitude_damping_lindblad_plus_state(t, gamma=0.02):
    """
    Analytical solution for amplitude damping starting from |+⟩ state
    
    The plus state density matrix is:
    ρ(0) = |+⟩⟨+| = [[0.5, 0.5], [0.5, 0.5]]
    
    Under amplitude damping, the evolution gives:
    ⟨σₓ⟩(t) = exp(-γt/2) * cos(Ωt)  [if there's a detuning Ω, otherwise just exp(-γt/2)]
    ⟨σᵧ⟩(t) = exp(-γt/2) * sin(Ωt)  [for pure damping, this becomes 0]
    ⟨σᵧ⟩(t) = (exp(-γt) - 1) / 2
    """
    # For pure amplitude damping (no Hamiltonian evolution)
    sigma_x = np.exp(-gamma * t / 2)  # Decays from initial value of 1
    sigma_y = np.zeros_like(t)        # Remains 0 for pure damping
    sigma_z = (np.exp(-gamma * t) - 1) / 2  # Evolves from 0 to -0.5
    
    return np.stack([sigma_x, sigma_y, sigma_z], axis=-1)

def load_model_components():
    """Load the trained model components"""
    # Model parameters (matching the saved models)
    latent_dim = 6
    nhidden = 53
    obs_dim = 3
    
    # Initialize model components
    func = LatentODEfunc(latent_dim, nhidden)
    rec = RecognitionRNN(latent_dim, obs_dim, nhidden, nbatch=1)
    dec = Decoder(latent_dim, obs_dim, nhidden)
    
    # Load saved weights
    try:
        func.load_state_dict(torch.load('saved_models/open_6_53_53_0.007_func.pt', map_location='cpu'))
        rec.load_state_dict(torch.load('saved_models/open_6_53_53_0.007_rec.pt', map_location='cpu'))
        dec.load_state_dict(torch.load('saved_models/open_6_53_53_0.007_dec.pt', map_location='cpu'))
        print("✓ Successfully loaded trained model weights")
    except FileNotFoundError as e:
        print(f"Warning: Could not load model weights - {e}")
        print("Using randomly initialized models for demonstration")
    
    return func, rec, dec

def generate_plus_state_trajectory(func, rec, dec, time_points):
    """Generate Neural ODE trajectory starting from plus state"""
    
    # Create plus state observables as initial condition
    # |+⟩ state has Pauli expectation values: ⟨σₓ⟩=1, ⟨σᵧ⟩=0, ⟨σᵧ⟩=0
    plus_state_observables = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    
    # Use RNN to encode initial condition (just one time step)
    rec.eval()
    with torch.no_grad():
        h = rec.initHidden()
        if h.shape[0] != 1:
            h = h[:1]  # Adjust batch size if needed
        
        qz0_mean, qz0_logvar = rec(plus_state_observables, h)[0].chunk(2, dim=1)
    
    # Sample from the latent distribution (or use mean for deterministic)
    epsilon = torch.randn(qz0_mean.size())
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
    
    # Integrate the Neural ODE
    func.eval()
    with torch.no_grad():
        pred_z = odeint(func, z0, time_points).squeeze()
        
        # Decode latent trajectory to observables
        if pred_z.dim() == 1:
            pred_z = pred_z.unsqueeze(0)
        
        pred_x = dec(pred_z)
    
    return pred_x.numpy()

def create_comparison_plot():
    """Create comprehensive comparison plot"""
    
    print("Loading trained Neural ODE models...")
    func, rec, dec = load_model_components()
    
    # Time evolution parameters
    t_max = 25.0
    n_points = 200
    gamma = 0.02
    
    time_points = torch.linspace(0, t_max, n_points)
    t_numpy = time_points.numpy()
    
    print("Generating theoretical Lindblad evolution...")
    theoretical_data = amplitude_damping_lindblad_plus_state(t_numpy, gamma)
    
    print("Generating Neural ODE predictions...")
    neural_predictions = generate_plus_state_trajectory(func, rec, dec, time_points)
    
    # Create the comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Neural ODE vs Lindblad Evolution: Plus State Amplitude Damping\n' + 
                 r'Initial State: $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$, $\gamma = 0.02$',
                 fontsize=16, y=0.95)
    
    pauli_labels = [r'$\langle\sigma_x\rangle$', r'$\langle\sigma_y\rangle$', r'$\langle\sigma_z\rangle$']
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    # Plot individual Pauli expectation values
    for i in range(3):
        ax = axes[i//2, i%2]
        
        # Theoretical curve
        ax.plot(t_numpy, theoretical_data[:, i], 
                color=colors[i], linewidth=3, linestyle='-', 
                label='Lindblad (Theoretical)', alpha=0.8)
        
        # Neural ODE predictions
        if neural_predictions.shape[0] == n_points:
            ax.plot(t_numpy, neural_predictions[:, i], 
                    color=colors[i], linewidth=2, linestyle='--', 
                    label='Neural ODE', alpha=0.9)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(pauli_labels[i])
        ax.set_title(f'Pauli {pauli_labels[i][8:-9]} Expectation Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-limits
        if i == 0:  # σₓ
            ax.set_ylim(-0.1, 1.1)
        elif i == 1:  # σᵧ  
            ax.set_ylim(-0.5, 0.5)
        else:  # σᵧ
            ax.set_ylim(-0.6, 0.1)
    
    # Combined plot in the fourth subplot
    ax = axes[1, 1]
    for i in range(3):
        ax.plot(t_numpy, theoretical_data[:, i], 
                color=colors[i], linewidth=2, linestyle='-', 
                label=f'{pauli_labels[i]} (Theory)', alpha=0.7)
        
        if neural_predictions.shape[0] == n_points:
            ax.plot(t_numpy, neural_predictions[:, i], 
                    color=colors[i], linewidth=2, linestyle='--', 
                    label=f'{pauli_labels[i]} (Neural)', alpha=0.9)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Expectation Value')
    ax.set_title('All Pauli Expectation Values')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = 'pauli_expectation_plus_state.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ Plot saved as {output_filename}")
    
    # Calculate and display metrics if Neural ODE data is available
    if neural_predictions.shape[0] == n_points:
        print("\n" + "="*60)
        print("PERFORMANCE METRICS - Plus State Initial Condition")
        print("="*60)
        
        for i, label in enumerate(['σₓ', 'σᵧ', 'σᵧ']):
            mse = np.mean((theoretical_data[:, i] - neural_predictions[:, i])**2)
            mae = np.mean(np.abs(theoretical_data[:, i] - neural_predictions[:, i]))
            
            # Correlation coefficient
            if np.std(theoretical_data[:, i]) > 1e-6 and np.std(neural_predictions[:, i]) > 1e-6:
                corr = np.corrcoef(theoretical_data[:, i], neural_predictions[:, i])[0, 1]
            else:
                corr = np.nan
            
            print(f"{label:>3}: MSE = {mse:.6f}, MAE = {mae:.6f}, Corr = {corr:.6f}")
        
        overall_mse = np.mean((theoretical_data - neural_predictions)**2)
        overall_mae = np.mean(np.abs(theoretical_data - neural_predictions))
        print(f"\nOverall: MSE = {overall_mse:.6f}, MAE = {overall_mae:.6f}")
        print("="*60)
    
    plt.show()
    return output_filename

if __name__ == "__main__":
    print("Neural ODE vs Lindblad Evolution - Plus State Analysis")
    print("="*55)
    
    output_file = create_comparison_plot()
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ Initial state: |+⟩ = (|0⟩ + |1⟩)/√2")
    print(f"✓ Damping rate: γ = 0.02")
    print("\nThis analysis demonstrates Neural ODE learning of quantum")
    print("amplitude damping dynamics starting from the plus state.")
