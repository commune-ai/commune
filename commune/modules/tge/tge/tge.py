import commune as c
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

class tge:
    def plots(self):
        """Generate all TGE visualization figures"""
        figures = {}
        
        # Distribution Pie Chart
        fig_dist = go.Figure(data=[go.Pie(
            labels=['Airdrop Claimable', 'Reserved', 'Future Allocation'],
            values=[1000000, 0, 0],
            hole=0.3,
            marker_colors=['#00ff00', '#ffaa00', '#0088ff']
        )])
        fig_dist.update_layout(title='$MOD Token Distribution (Phase 1)', showlegend=True)
        figures['distribution'] = fig_dist
        
        # Timeline Chart
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=[datetime(2025, 10, 15), datetime(2026, 1, 1)],
            y=[1000000, 1000000],
            mode='lines+markers',
            name='Total Supply',
            line=dict(color='#00ff00', width=3),
            marker=dict(size=10)
        ))
        fig_time.update_layout(title='Phase 1 Timeline', xaxis_title='Date', yaxis_title='Tokens Available', hovermode='x unified')
        figures['timeline'] = fig_time
        
        # Progress Gauge
        fig_prog = go.Figure(go.Indicator(
            mode='gauge+number',
            value=0,
            title={'text': 'Claim Progress'},
            gauge={'axis': {'range': [None, 1000000]}, 'bar': {'color': '#00ff00'}}
        ))
        figures['progress'] = fig_prog
        
        # Lock Multiplier Curve (0x -> 4x)
        lock_days = np.linspace(0, 730, 100)
        multipliers = np.where(lock_days == 0, 0, (lock_days / 365) ** 2)
        
        fig_lock = go.Figure()
        fig_lock.add_trace(go.Scatter(
            x=lock_days/365,
            y=multipliers,
            mode='lines',
            name='Lock Multiplier',
            line=dict(color='#00ff00', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)'
        ))
        fig_lock.add_trace(go.Scatter(
            x=[0, 1, 2],
            y=[0, 1, 4],
            mode='markers+text',
            name='Key Points',
            marker=dict(size=12, color='#ff0000'),
            text=['0x (No Lock)', '1x (1 Year)', '4x (2 Years)'],
            textposition='top center'
        ))
        fig_lock.update_layout(
            title='Phase 2: Lock Time Multiplier Curve',
            xaxis_title='Lock Period (Years)',
            yaxis_title='Multiplier',
            hovermode='x unified',
            showlegend=True
        )
        figures['lock_multiplier'] = fig_lock
        
        # Vesting Schedule
        fig_vest = go.Figure()
        for period in [365, 730]:
            days = np.linspace(0, period, 100)
            mult = (period / 365) ** 2
            vested = (days / period) * 1000000 * mult
            fig_vest.add_trace(go.Scatter(
                x=days,
                y=vested,
                mode='lines',
                name=f'{period//365} Year Lock ({mult:.1f}x)',
                line=dict(width=3)
            ))
        fig_vest.update_layout(
            title='Phase 2: Vesting Schedule (Linear Distribution)',
            xaxis_title='Days Since Lock',
            yaxis_title='Tokens Vested',
            hovermode='x unified',
            showlegend=True
        )
        figures['vesting'] = fig_vest
        
        return figures
    
    def show_plots(self):
        """Display all plots interactively"""
        figures = self.plots()
        for name, fig in figures.items():
            fig.show()
        return figures
    
    def save_images(self, output_dir='public'):
        """Save plots as PNG images"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        figures = self.plots()
        for name, fig in figures.items():
            fig.write_image(f'{output_dir}/{name}.png', width=1200, height=650)
        return f'Images saved to {output_dir}/'
