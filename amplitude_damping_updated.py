"""
Amplitude Damping Channel: Updated Implementation
================================================

This script implements amplitude damping using the analytical approach
consistent with the existing codebase style and structure.

Author: Serah
Date: 2025-08-04
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import seaborn as sns

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AmplitudeDampingKraus:
    """
    Analytical implementation of amplitude damping using Kraus operators
    """
    
    @staticmethod
    def kraus_operators(gamma, t):
        """
        Kraus operators for amplitude damping channel
        
        Args:
            gamma: Damping rate
            t: Time
            
        Returns:
            K0, K1: Kraus operators as numpy arrays
        """
        p = 1 - np.exp(-gamma * t)
        K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
        K1 = np.array([[0, np.sqrt(p)], [0, 0]])
        return K0, K1
    
    @staticmethod
    def evolve_density_matrix(rho_0, gamma, t):
        """
        Analytically evolve density matrix under amplitude damping
        
        Args:
            rho_0: Initial density matrix (numpy array)
            gamma: Damping rate
            t: Time
            
        Returns:
            rho_t: Density matrix at time t
        """
        K0, K1 = AmplitudeDampingKraus.kraus_operators(gamma, t)
        rho_t = K0 @ rho_0 @ K0.conj().T + K1 @ rho_0 @ K1.conj().T
        return rho_t
    
    @staticmethod
    def analytical_expectation_values(initial_state, gamma, times):
        """
        Calculate analytical expectation values for Pauli operators
        
        Args:
            initial_state: Initial quantum state (QuTiP ket)
            gamma: Damping rate
            times: Array of time points
            
        Returns:
            expectations: Array of [<σx>, <σy>, <σz>] over time
        """
        # Convert initial state to density matrix
        rho_0 = initial_state * initial_state.dag()
        rho_0_array = rho_0.full()
        
        expectations = []
        for t in times:
            # Evolve density matrix
            rho_t = AmplitudeDampingKraus.evolve_density_matrix(rho_0_array, gamma, t)
            
            # Calculate expectation values
            exp_x = np.real(np.trace(sigmax().full() @ rho_t))
            exp_y = np.real(np.trace(sigmay().full() @ rho_t))
            exp_z = np.real(np.trace(sigmaz().full() @ rho_t))
            
            expectations.append([exp_x, exp_y, exp_z])
        
        return np.array(expectations)

class AmplitudeDampingAnalysis:
    """
    Analysis class for amplitude damping evolution
    """
    
    def __init__(self, gamma=0.02, time_end=10.0, num_points=200):
        """
        Initialize the amplitude damping analysis
        
        Args:
            gamma: Amplitude damping rate
            time_end: End time for analysis
            num_points: Number of time points
        """
        self.gamma = gamma
        self.time_end = time_end
        self.num_points = num_points
        self.time_points = np.linspace(0, time_end, num_points)
        
    def compare_methods(self, initial_state):
        """
        Compare analytical Kraus approach with numerical mesolve
        
        Args:
            initial_state: Initial quantum state
            
        Returns:
            Dictionary with analytical and numerical results
        """
        # Analytical solution using Kraus operators
        analytical_result = AmplitudeDampingKraus.analytical_expectation_values(
            initial_state, self.gamma, self.time_points
        )
        
        # Numerical solution using mesolve
        c_ops = [np.sqrt(self.gamma) * destroy(2)]
        numerical_result = mesolve(
            0 * sigmaz(), initial_state, self.time_points,
            c_ops=c_ops, e_ops=[sigmax(), sigmay(), sigmaz()],
            progress_bar=None
        )
        numerical_result = np.array(numerical_result.expect).T
        
        return {
            'analytical': analytical_result,
            'numerical': numerical_result
        }
    
    def create_plots(self, initial_state):
        """
        Create comprehensive plots for amplitude damping analysis
        
        Args:
            initial_state: Initial quantum state
        """
        results = self.compare_methods(initial_state)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Amplitude Damping Analysis: Analytical vs Numerical', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: All expectation values (analytical)
        ax = axes[0, 0]
        ax.plot(self.time_points, results['analytical'][:, 0], 'r-', 
               linewidth=2.5, label=r'$\langle\sigma_x\rangle$', marker='o', markersize=2)
        ax.plot(self.time_points, results['analytical'][:, 1], 'g-', 
               linewidth=2.5, label=r'$\langle\sigma_y\rangle$', marker='s', markersize=2)
        ax.plot(self.time_points, results['analytical'][:, 2], 'b-', 
               linewidth=2.5, label=r'$\langle\sigma_z\rangle$', marker='^', markersize=2)
        ax.set_title('Analytical Solution (Kraus Operators)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Expectation Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: All expectation values (numerical)
        ax = axes[0, 1]
        ax.plot(self.time_points, results['numerical'][:, 0], 'r--', 
               linewidth=2, label=r'$\langle\sigma_x\rangle$')
        ax.plot(self.time_points, results['numerical'][:, 1], 'g--', 
               linewidth=2, label=r'$\langle\sigma_y\rangle$')
        ax.plot(self.time_points, results['numerical'][:, 2], 'b--', 
               linewidth=2, label=r'$\langle\sigma_z\rangle$')
        ax.set_title('Numerical Solution (mesolve)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Expectation Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Comparison for σz (most important for amplitude damping)
        ax = axes[1, 0]
        ax.plot(self.time_points, results['analytical'][:, 2], 'b-', 
               linewidth=3, label='Analytical', alpha=0.8)
        ax.plot(self.time_points, results['numerical'][:, 2], 'r--', 
               linewidth=2, label='Numerical', alpha=0.8)
        
        # Add theoretical exponential decay for |1⟩ initial state
        if np.abs(expect(sigmaz(), initial_state) - 1) < 0.01:  # Check if initial state is |1⟩
            theory_envelope = np.exp(-self.gamma * self.time_points)
            ax.plot(self.time_points, theory_envelope, 'k:', linewidth=1.5, 
                   label=r'$e^{-\gamma t}$ envelope', alpha=0.6)
        
        ax.set_title(r'Population Evolution: $\langle\sigma_z\rangle$')
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$\langle\sigma_z\rangle$')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate and display error metrics
        mse = np.mean((results['analytical'][:, 2] - results['numerical'][:, 2])**2)
        ax.text(0.6, 0.9, f'MSE: {mse:.2e}', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        # Plot 4: Error analysis
        ax = axes[1, 1]
        errors = results['analytical'] - results['numerical']
        ax.plot(self.time_points, errors[:, 0], 'r-', label=r'Error in $\langle\sigma_x\rangle$')
        ax.plot(self.time_points, errors[:, 1], 'g-', label=r'Error in $\langle\sigma_y\rangle$')
        ax.plot(self.time_points, errors[:, 2], 'b-', label=r'Error in $\langle\sigma_z\rangle$')
        ax.set_title('Analytical - Numerical Errors')
        ax.set_xlabel('Time')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text box with initial state info
        state_info = f'Initial State: |1⟩\nγ = {self.gamma}\nAmplitude Damping'
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        axes[0, 0].text(0.02, 0.98, state_info, transform=axes[0, 0].transAxes, 
                       fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('amplitude_damping_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results

def main():
    """
    Main execution function
    """
    print("="*60)
    print("AMPLITUDE DAMPING ANALYSIS")
    print("Analytical (Kraus) vs Numerical (mesolve)")
    print("="*60)
    
    # Parameters
    gamma = 0.02   # decay rate
    
    # Initial state: |ψ⟩ = |1⟩ (excited state)
    psi0 = basis(2, 1)
    
    # Initialize analysis
    analysis = AmplitudeDampingAnalysis(gamma=gamma, time_end=10.0, num_points=200)
    
    # Run analysis and create plots
    results = analysis.create_plots(psi0)
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Initial state: |1⟩")
    print(f"- Damping rate γ = {gamma}")
    print(f"- Time range: 0 to {analysis.time_end}")
    print(f"- Number of points: {analysis.num_points}")
    
    analytical = results['analytical']
    numerical = results['numerical']
    
    print(f"\nAt t=0:")
    print(f"  ⟨σx⟩ = {analytical[0, 0]:.3f} (analytical), {numerical[0, 0]:.3f} (numerical)")
    print(f"  ⟨σy⟩ = {analytical[0, 1]:.3f} (analytical), {numerical[0, 1]:.3f} (numerical)")
    print(f"  ⟨σz⟩ = {analytical[0, 2]:.3f} (analytical), {numerical[0, 2]:.3f} (numerical)")
    print(f"  Note: For |1⟩ state, ⟨σz⟩ = +1, but QuTiP uses -1 convention")
    
    print(f"\nAt t={analysis.time_end}:")
    print(f"  ⟨σx⟩ = {analytical[-1, 0]:.3f} (analytical), {numerical[-1, 0]:.3f} (numerical)")
    print(f"  ⟨σy⟩ = {analytical[-1, 1]:.3f} (analytical), {numerical[-1, 1]:.3f} (numerical)")
    print(f"  ⟨σz⟩ = {analytical[-1, 2]:.3f} (analytical), {numerical[-1, 2]:.3f} (numerical)")
    
    # Error analysis
    mse_total = np.mean((analytical - numerical)**2)
    print(f"\nOverall Mean Squared Error: {mse_total:.2e}")
    
    print("\n✅ Analysis complete! Plot saved as 'amplitude_damping_analysis.png'")

if __name__ == "__main__":
    main()
