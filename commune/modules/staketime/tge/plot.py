import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class VestingMultiplierPlotter:
    """Creates combined and separate plots for vesting schedule and multipliers"""
    
    def __init__(self):
        self.total_days = 730  # 2 years
        self.anchors = [
            {"day": 0, "multiplier": 0},
            {"day": 365, "multiplier": 1},
            {"day": 730, "multiplier": 4}
        ]
    
    def calculate_vesting_percentage(self, day):
        """Calculate linear vesting percentage"""
        return min(100, (day / self.total_days) * 100)
    
    def calculate_multiplier(self, day):
        """Calculate exponential multiplier"""
        t = day / 365  # Convert to years
        if t == 0:
            return 0
        elif t <= 1:
            return t  # Linear for first year
        else:
            return 4 ** (t - 1)  # Exponential after year 1
    
    def plot_separate(self):
        """Create separate plots for vesting and multiplier curves"""
        days = np.linspace(0, self.total_days, 1000)
        vesting = [self.calculate_vesting_percentage(d) for d in days]
        multipliers = [self.calculate_multiplier(d) for d in days]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Vesting plot
        ax1.plot(days, vesting, 'b-', linewidth=2.5, label='Linear Vesting')
        ax1.fill_between(days, 0, vesting, alpha=0.3, color='blue')
        ax1.set_ylabel('Vested Percentage (%)', fontsize=12)
        ax1.set_title('Token Vesting Schedule (Linear over 2 Years)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right')
        ax1.set_xlim(0, self.total_days)
        ax1.set_ylim(0, 105)
        
        # Multiplier plot
        ax2.plot(days, multipliers, 'g-', linewidth=2.5, label='Exponential Multiplier')
        ax2.fill_between(days, 0, multipliers, alpha=0.3, color='green')
        ax2.set_xlabel('Days from TGE', fontsize=12)
        ax2.set_ylabel('Multiplier (x)', fontsize=12)
        ax2.set_title('StakeTime Multiplier Curve (Exponential Growth)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        ax2.set_xlim(0, self.total_days)
        ax2.set_ylim(-0.5, 4.5)
        
        # Add milestone markers to both plots
        milestones = [0, 90, 180, 365, 540, 730]
        milestone_labels = ['TGE', '3mo', '6mo', '1yr', '18mo', '2yr']
        
        for day, label in zip(milestones, milestone_labels):
            vested = self.calculate_vesting_percentage(day)
            mult = self.calculate_multiplier(day)
            
            ax1.plot(day, vested, 'ro', markersize=8)
            ax1.annotate(f'{label}\n{vested:.1f}%', xy=(day, vested), 
                        xytext=(day, vested + 5), ha='center', fontsize=9)
            
            ax2.plot(day, mult, 'ro', markersize=8)
            ax2.annotate(f'{label}\n{mult:.2f}x', xy=(day, mult),
                        xytext=(day, mult + 0.3), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('separate_plots.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Separate plots saved to separate_plots.png")
    
    def plot_combined(self):
        """Create combined plot with dual y-axes"""
        days = np.linspace(0, self.total_days, 1000)
        vesting = [self.calculate_vesting_percentage(d) for d in days]
        multipliers = [self.calculate_multiplier(d) for d in days]
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Primary axis - Vesting
        color = 'tab:blue'
        ax1.set_xlabel('Days from TGE', fontsize=12)
        ax1.set_ylabel('Vested Percentage (%)', color=color, fontsize=12)
        line1 = ax1.plot(days, vesting, color=color, linewidth=2.5, label='Linear Vesting')
        ax1.fill_between(days, 0, vesting, alpha=0.2, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)
        
        # Secondary axis - Multiplier
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Multiplier (x)', color=color, fontsize=12)
        line2 = ax2.plot(days, multipliers, color=color, linewidth=2.5, label='Exponential Multiplier')
        ax2.fill_between(days, 0, multipliers, alpha=0.2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.5, 4.5)
        
        # Add milestone markers
        milestones = [0, 90, 180, 365, 540, 730]
        milestone_labels = ['TGE', '3mo', '6mo', '1yr', '18mo', '2yr']
        
        for day, label in zip(milestones, milestone_labels):
            vested = self.calculate_vesting_percentage(day)
            mult = self.calculate_multiplier(day)
            ax1.plot(day, vested, 'bo', markersize=8)
            ax2.plot(day, mult, 'go', markersize=8)
            ax1.axvline(x=day, color='gray', linestyle='--', alpha=0.3)
            ax1.text(day, -8, label, ha='center', fontsize=9)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', framealpha=0.9)
        
        plt.title('TGE Vesting Schedule with StakeTime Multipliers', fontsize=16, fontweight='bold', pad=20)
        fig.tight_layout()
        plt.savefig('combined_plot.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Combined plot saved to combined_plot.png")
    
    def generate_all_plots(self):
        """Generate both separate and combined plots"""
        print("\n=== Generating Vesting and Multiplier Plots ===")
        print("\n1. Creating separate plots...")
        self.plot_separate()
        print("\n2. Creating combined plot...")
        self.plot_combined()
        print("\n✅ All plots generated successfully!")
        print("\nKey insights:")
        print("- Vesting: Linear from 0% to 100% over 730 days")
        print("- Multiplier: Linear 0x to 1x (Year 1), then exponential to 4x (Year 2)")
        print("- Separate plots provide focused view of each metric")
        print("- Combined plot shows correlation between vesting and rewards")

if __name__ == "__main__":
    plotter = VestingMultiplierPlotter()
    plotter.generate_all_plots()
