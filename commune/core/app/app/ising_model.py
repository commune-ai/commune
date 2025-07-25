import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

class IsingModel:
    """
    2D Ising Model implementation with Metropolis algorithm
    """
    
    def __init__(self, size: int = 50, temperature: float = 2.0, J: float = 1.0):
        """
        Initialize the Ising model
        
        Args:
            size: Grid size (NxN)
            temperature: System temperature (in units of J/k_B)
            J: Coupling constant
        """
        self.size = size
        self.temperature = temperature
        self.J = J
        self.beta = 1.0 / temperature if temperature > 0 else float('inf')
        
        # Initialize random spin configuration (+1 or -1)
        self.spins = np.random.choice([-1, 1], size=(size, size))
        
        # Precompute energy differences for efficiency
        self._precompute_boltzmann_factors()
        
    def _precompute_boltzmann_factors(self):
        """Precompute Boltzmann factors for possible energy changes"""
        self.boltzmann_factors = {}
        for delta_E in [-8, -4, 0, 4, 8]:
            self.boltzmann_factors[delta_E] = np.exp(-self.beta * delta_E)
    
    def _calculate_site_energy(self, i: int, j: int) -> float:
        """Calculate energy contribution from a single site"""
        spin = self.spins[i, j]
        neighbors_sum = (
            self.spins[(i+1) % self.size, j] +
            self.spins[(i-1) % self.size, j] +
            self.spins[i, (j+1) % self.size] +
            self.spins[i, (j-1) % self.size]
        )
        return -self.J * spin * neighbors_sum
    
    def calculate_total_energy(self) -> float:
        """Calculate total energy of the system"""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                energy += self._calculate_site_energy(i, j)
        return energy / 2  # Divide by 2 to avoid double counting
    
    def calculate_magnetization(self) -> float:
        """Calculate magnetization per spin"""
        return np.mean(self.spins)
    
    def metropolis_step(self):
        """Perform one Metropolis Monte Carlo step"""
        # Choose random site
        i = np.random.randint(0, self.size)
        j = np.random.randint(0, self.size)
        
        # Calculate energy change if spin is flipped
        spin = self.spins[i, j]
        neighbors_sum = (
            self.spins[(i+1) % self.size, j] +
            self.spins[(i-1) % self.size, j] +
            self.spins[i, (j+1) % self.size] +
            self.spins[i, (j-1) % self.size]
        )
        delta_E = 2 * self.J * spin * neighbors_sum
        
        # Accept or reject the flip
        if delta_E <= 0 or np.random.random() < self.boltzmann_factors.get(delta_E, np.exp(-self.beta * delta_E)):
            self.spins[i, j] *= -1
    
    def simulate(self, steps: int, measure_interval: int = 100) -> Tuple[list, list]:
        """Run simulation for given number of steps"""
        energies = []
        magnetizations = []
        
        for step in range(steps):
            self.metropolis_step()
            
            if step % measure_interval == 0:
                energies.append(self.calculate_total_energy() / (self.size**2))
                magnetizations.append(abs(self.calculate_magnetization()))
        
        return energies, magnetizations
    
    def visualize(self, save_path: Optional[str] = None):
        """Visualize the current spin configuration"""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.spins, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Spin')
        plt.title(f'Ising Model (T={self.temperature:.2f}, Size={self.size}x{self.size})')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


class IsingAPI:
    """API wrapper for Ising model simulations"""
    
    def __init__(self):
        self.models = {}
        self.results_dir = './results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def create_model(self, name: str, size: int = 50, temperature: float = 2.0) -> dict:
        """Create a new Ising model instance"""
        self.models[name] = IsingModel(size=size, temperature=temperature)
        return {
            'success': True,
            'message': f'Model {name} created',
            'config': {
                'name': name,
                'size': size,
                'temperature': temperature
            }
        }
    
    def simulate(self, name: str, steps: int = 10000, measure_interval: int = 100) -> dict:
        """Run simulation on a model"""
        if name not in self.models:
            return {'success': False, 'error': f'Model {name} not found'}
        
        model = self.models[name]
        energies, magnetizations = model.simulate(steps, measure_interval)
        
        # Save results
        results = {
            'energies': energies,
            'magnetizations': magnetizations,
            'temperature': model.temperature,
            'size': model.size,
            'steps': steps
        }
        
        return {
            'success': True,
            'results': results,
            'stats': {
                'mean_energy': np.mean(energies),
                'mean_magnetization': np.mean(magnetizations),
                'std_energy': np.std(energies),
                'std_magnetization': np.std(magnetizations)
            }
        }
    
    def get_state(self, name: str) -> dict:
        """Get current state of a model"""
        if name not in self.models:
            return {'success': False, 'error': f'Model {name} not found'}
        
        model = self.models[name]
        return {
            'success': True,
            'state': {
                'spins': model.spins.tolist(),
                'energy': model.calculate_total_energy(),
                'magnetization': model.calculate_magnetization(),
                'temperature': model.temperature
            }
        }
    
    def visualize_model(self, name: str, save: bool = True) -> dict:
        """Visualize a model's current state"""
        if name not in self.models:
            return {'success': False, 'error': f'Model {name} not found'}
        
        save_path = os.path.join(self.results_dir, f'{name}_state.png') if save else None
        self.models[name].visualize(save_path)
        
        return {
            'success': True,
            'message': f'Visualization saved to {save_path}' if save else 'Visualization displayed'
        }
    
    def phase_transition_study(self, size: int = 30, temp_range: Tuple[float, float] = (1.0, 4.0), 
                             temp_steps: int = 20, mc_steps: int = 5000) -> dict:
        """Study phase transition by varying temperature"""
        temperatures = np.linspace(temp_range[0], temp_range[1], temp_steps)
        magnetizations = []
        energies = []
        
        for T in temperatures:
            model = IsingModel(size=size, temperature=T)
            # Equilibrate
            for _ in range(mc_steps // 2):
                model.metropolis_step()
            # Measure
            E, M = model.simulate(mc_steps // 2, measure_interval=10)
            magnetizations.append(np.mean(M))
            energies.append(np.mean(E))
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(temperatures, magnetizations, 'bo-')
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Magnetization')
        ax1.set_title('Phase Transition in 2D Ising Model')
        ax1.axvline(x=2.269, color='r', linestyle='--', label='Critical Temperature')
        ax1.legend()
        
        ax2.plot(temperatures, energies, 'ro-')
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Energy per spin')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'phase_transition.png')
        plt.savefig(save_path)
        plt.close()
        
        return {
            'success': True,
            'results': {
                'temperatures': temperatures.tolist(),
                'magnetizations': magnetizations,
                'energies': energies,
                'critical_temp_theoretical': 2.269,
                'plot_saved': save_path
            }
        }


# Example usage and testing
if __name__ == '__main__':
    # Create API instance
    api = IsingAPI()
    
    # Create a model
    print(api.create_model('test_model', size=50, temperature=2.0))
    
    # Run simulation
    results = api.simulate('test_model', steps=10000)
    print(f"Simulation stats: {results['stats']}")
    
    # Visualize
    api.visualize_model('test_model')
    
    # Phase transition study
    print("Running phase transition study...")
    phase_results = api.phase_transition_study()
    print(f"Phase transition study completed. Results saved to {phase_results['results']['plot_saved']}")
