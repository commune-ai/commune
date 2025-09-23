import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('TGE Analytics Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Token Distribution
labels = ['Team & Advisors (15%)', 'Community Rewards (30%)', 'Liquidity & Market Making (20%)', 
          'Development Fund (15%)', 'Strategic Partnerships (10%)', 'Public Sale (10%)']
sizes = [15, 30, 20, 15, 10, 10]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#DDA0DD']
ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title('Token Distribution Breakdown')

# Plot 2: Vesting Schedule
months = np.arange(0, 49, 1)
team_vesting = np.where(months < 12, 0, np.minimum((months - 12) / 36 * 100, 100))
community_vesting = np.minimum(months / 24 * 100, 100)
liquidity_vesting = np.where(months < 3, months / 3 * 100, 100)

ax2.plot(months, team_vesting, label='Team (12mo cliff, 36mo vest)', linewidth=2)
ax2.plot(months, community_vesting, label='Community (24mo linear)', linewidth=2)
ax2.plot(months, liquidity_vesting, label='Liquidity (3mo unlock)', linewidth=2)
ax2.set_xlabel('Months from TGE')
ax2.set_ylabel('Tokens Unlocked (%)')
ax2.set_title('Vesting Schedule Timeline')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Price Projection
days = np.arange(0, 365)
price_base = 0.10
price_projection = price_base * (1 + 0.15 * np.sin(days/30) + days/365 * 2)
price_lower = price_projection * 0.7
price_upper = price_projection * 1.3

ax3.plot(days, price_projection, 'b-', label='Expected Price', linewidth=2)
ax3.fill_between(days, price_lower, price_upper, alpha=0.2, label='Confidence Band')
ax3.set_xlabel('Days from TGE')
ax3.set_ylabel('Token Price ($)')
ax3.set_title('Price Projection (First Year)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Market Cap Growth
market_cap_data = [100, 250, 500, 750, 1000, 1500, 2000, 2500]
quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
colors_bar = ['#3498db' if i < 4 else '#2ecc71' for i in range(len(quarters))]

ax4.bar(quarters, market_cap_data, color=colors_bar)
ax4.set_xlabel('Quarter')
ax4.set_ylabel('Market Cap ($M)')
ax4.set_title('Projected Market Cap Growth')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('tge_analytics.png', dpi=300, bbox_inches='tight')
plt.show()

print('TGE Analytics plots saved as tge_analytics.png')