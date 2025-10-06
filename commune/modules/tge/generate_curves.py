import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('public', exist_ok=True)

# Lock Multiplier Curve (0x -> 4x) - SAME AS MULTIPLIER CURVE
lock_days = np.linspace(0, 730, 100)
multipliers = np.where(lock_days == 0, 0, (lock_days / 365) ** 2)

plt.figure(figsize=(12, 6))
plt.plot(lock_days/365, multipliers, 'g-', linewidth=3, label='Lock Curve = Multiplier Curve')
plt.fill_between(lock_days/365, multipliers, alpha=0.2, color='green')
plt.scatter([0, 1, 2], [0, 1, 4], color='red', s=150, zorder=5, label='Key Points')
plt.text(0, 0.2, '0x (No Lock)', ha='center', fontsize=10)
plt.text(1, 1.2, '1x (1 Year)', ha='center', fontsize=10)
plt.text(2, 4.2, '4x (2 Years)', ha='center', fontsize=10)
plt.xlabel('Lock Period (Years)', fontsize=12)
plt.ylabel('Multiplier', fontsize=12)
plt.title('Phase 2: Lock Curve = Multiplier Curve (Unified)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('public/lock_multiplier.png', dpi=150, bbox_inches='tight')
plt.close()

# Vesting Curve
plt.figure(figsize=(12, 6))
colors = ['#1f77b4', '#ff7f0e']
for i, period in enumerate([365, 730]):
    days = np.linspace(0, period, 100)
    mult = (period / 365) ** 2
    vested = (days / period) * 1000000 * mult
    plt.plot(days, vested, linewidth=3, color=colors[i], label=f'{period//365} Year Lock ({mult:.1f}x)')

plt.xlabel('Days Since Lock', fontsize=12)
plt.ylabel('Tokens Vested', fontsize=12)
plt.title('Phase 2: Vesting Schedule (Linear Distribution)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('public/vesting_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print('✓ Generated lock_multiplier.png (UNIFIED: Lock Curve = Multiplier Curve)')
print('✓ Generated vesting_curve.png')
print('✓ All curves saved to public/ directory')