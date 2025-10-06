import commune as c
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

class tge:

    def plots(self, *args, **kwargs):
        """Generate interactive figures for token distribution visualization"""
        figures = {}
        
        # Token Distribution Pie Chart
        distribution_data = {
            'Category': ['Airdrop Claimable', 'Reserved', 'Future Allocation'],
            'Tokens': [1000000, 0, 0]
        }
        
        fig_distribution = go.Figure(data=[go.Pie(
            labels=distribution_data['Category'],
            values=distribution_data['Tokens'],
            hole=0.3,
            marker_colors=['#00ff00', '#ffaa00', '#0088ff']
        )])
        fig_distribution.update_layout(
            title='$MOD Token Distribution (Phase 1)',
            showlegend=True
        )
        figures['distribution'] = fig_distribution
        
        # Timeline Chart
        start_date = datetime(2025, 10, 15)
        end_date = datetime(2026, 1, 1)
        
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=[start_date, end_date],
            y=[1000000, 1000000],
            mode='lines+markers',
            name='Total Supply',
            line=dict(color='#00ff00', width=3),
            marker=dict(size=10)
        ))
        fig_timeline.update_layout(
            title='Phase 1 Timeline: Token Distribution',
            xaxis_title='Date',
            yaxis_title='Tokens Available',
            hovermode='x unified'
        )
        figures['timeline'] = fig_timeline
        
        # Claim Progress Gauge
        fig_progress = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'Claim Progress'},
            gauge={
                'axis': {'range': [None, 1000000]},
                'bar': {'color': '#00ff00'},
                'steps': [
                    {'range': [0, 500000], 'color': '#333333'},
                    {'range': [500000, 1000000], 'color': '#555555'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 1000000
                }
            }
        ))
        figures['progress'] = fig_progress
        
        # Lock Multiplier Curve - EXPONENTIAL (0x at 0 days, 1x at 1 year, 4x at 2 years)
        lock_days = np.linspace(0, 730, 100)
        multipliers = np.where(lock_days == 0, 0, (lock_days / 365) ** 2)
        
        fig_lock_multiplier = go.Figure()
        fig_lock_multiplier.add_trace(go.Scatter(
            x=lock_days / 365,
            y=multipliers,
            mode='lines',
            name='Lock Multiplier',
            line=dict(color='#00ff00', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)'
        ))
        
        fig_lock_multiplier.add_trace(go.Scatter(
            x=[0, 1, 2],
            y=[0, 1, 4],
            mode='markers+text',
            name='Key Points',
            marker=dict(size=12, color='#ff0000'),
            text=['0x (No Lock)', '1x (1 Year)', '4x (2 Years)'],
            textposition='top center'
        ))
        
        fig_lock_multiplier.update_layout(
            title='Phase 2: Lock Time Multiplier Curve (Exponential)',
            xaxis_title='Lock Period (Years)',
            yaxis_title='Multiplier',
            hovermode='x unified',
            showlegend=True
        )
        figures['lock_multiplier'] = fig_lock_multiplier
        
        # Vesting Schedule - NOW EXPONENTIAL TO MATCH LOCK MULTIPLIER
        vesting_periods = [365, 730]
        fig_vesting = go.Figure()
        
        for period_days in vesting_periods:
            days = np.linspace(0, period_days, 100)
            multiplier = (period_days / 365) ** 2
            base_tokens = 1000000
            # EXPONENTIAL VESTING: (days/period)^2 * base * multiplier
            vested_tokens = ((days / period_days) ** 2) * base_tokens * multiplier
            
            fig_vesting.add_trace(go.Scatter(
                x=days,
                y=vested_tokens,
                mode='lines',
                name=f'{period_days // 365} Year Lock ({multiplier:.1f}x)',
                line=dict(width=3)
            ))
        
        fig_vesting.update_layout(
            title='Phase 2: Vesting Schedule (Exponential - Matches Lock Multiplier)',
            xaxis_title='Days Since Lock',
            yaxis_title='Tokens Vested',
            hovermode='x unified',
            showlegend=True
        )
        figures['vesting'] = fig_vesting
        
        return figures
    
    def show_plots(self):
        """Display all generated plots"""
        figures = self.plots()
        for name, fig in figures.items():
            fig.show()
        return figures