# Plan B: Stake-Time Multiplier Vesting Curve

import math
import numpy as np
import plotly.graph_objects as go
class PlanB:
    def __init__(self, total_supply=4_200_000, scale_factor=4, power=2):
        self.total_supply = total_supply
        self.max_lock_days = 730  # 2 years
        self.scale_factor = scale_factor
        self.power = power
        
    def calculate_multiplier(self, lock_days):
        """
        Calculate the multiplier based on lock duration.
        0 days = 0x, 365 days (1 year) = 1x, 730 days (2 years) = 4x
        """
        return ((lock_days / 365) ** self.power)/self.scale_factor
    
    def calculate_allocation(self, user_stake, user_lock_days, total_weighted_stakes):
        """
        Calculate user's token allocation based on stake and lock duration.
        """
        user_multiplier = self.calculate_multiplier(user_lock_days)
        user_weighted_stake = user_stake * user_multiplier
        
        if total_weighted_stakes == 0:
            return 0
            
        allocation = (user_weighted_stake / total_weighted_stakes) * self.total_supply
        return allocation

    def plot_multiplier(self, show_chart=True):
        """
        Display the Plan B vesting curve visualization.

        Args:
            years: Number of years to display (default: 2)
            show_chart: Whether to display the chart (default: True)
        """
        import plotly.graph_objects as go
        import numpy as np

        # Generate data points
        days = np.linspace(0, self.max_lock_days, 1000)
        multipliers = [self.calculate_multiplier(day) for day in days]

        # Create the figure
        fig = go.Figure()

        # Add the multiplier curve
        fig.add_trace(go.Scatter(
            x=days,
            y=multipliers,
            mode='lines',
            name='Stake Multiplier',
            line=dict(color='purple', width=3),
            hovertemplate='Days: %{x:.0f}<br>Multiplier: %{y:.2f}x<extra></extra>'
        ))

        # Add milestone markers
        milestones = [
            (0, 'No lock'),
            (183, '6 months'),
            (365, '1 year'),
            (548, '18 months'),
            (730, '2 years')
        ]

        for days_locked, label in milestones:
            multiplier = self.calculate_multiplier(days_locked)
            fig.add_trace(go.Scatter(
                x=[days_locked],
                y=[multiplier],
                mode='markers+text',
                name=label,
                marker=dict(size=12, color='red'),
                text=[f'{label}: {multiplier:.2f}x'],
                textposition='top center',
                showlegend=False
            ))

        # Update layout
        fig.update_layout(
            title={
                'text': 'Plan B: Stake-Time Multiplier Curve',
                'font': {'size': 24}
            },
            xaxis_title='Lock Duration (Days)',
            yaxis_title='Multiplier',
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=700,
            yaxis=dict(
                tickformat='.1f',
                gridcolor='lightgray',
                range=[0, 1.1]
            ),
            xaxis=dict(
                gridcolor='lightgray',
                range=[0, self.max_lock_days + 10]
            ),
            annotations=[
                dict(
                    x=365,
                    y=1,
                    xref='x',
                    yref='y',
                    text='1x at 1 year',
                    showarrow=True,
                    arrowhead=2,
                    ax=50,
                    ay=-30
                ),
                dict(
                    x=730,
                    y=1,
                    xref='x',
                    yref='y',
                    text='2 years',
                    showarrow=True,
                    arrowhead=2,
                    ax=-50,
                    ay=-30
                )
            ]
        )



        if show_chart:
            fig.show()

        return fig

    def plot_vesting(self, show_chart=True):
        """
        Display the Plan B vesting curve visualization.

        Args:
            show_chart: Whether to display the chart (default: True)
        """
        import plotly.graph_objects as go
        import numpy as np

        # Generate data points
        days = np.linspace(0, self.max_lock_days, 1000)
        vesting_values = [self.calculate_multiplier(day) for day in days]

        # Create the figure
        fig = go.Figure()

        # Add the vesting curve
        fig.add_trace(go.Scatter(
            x=days,
            y=vesting_values,
            mode='lines',
            name='Vesting Curve',
            line=dict(color='green', width=3),
            hovertemplate='Days: %{x:.0f}<br>Vesting Value: %{y:.2f}<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title={
                'text': 'Plan B: Vesting Curve',
                'font': {'size': 24}
            },
            xaxis_title='Lock Duration (Days)',
            yaxis_title='Vesting Value',
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=700,
            yaxis=dict(
                tickformat='.1f',
                gridcolor='lightgray',
                range=[0, 1.1]
            ),
            xaxis=dict(
                gridcolor='lightgray',
                range=[0, self.max_lock_days + 10]
            )
        )

        if show_chart:
            fig.show()

        return fig

    def plot(self, show_chart=True):
        """
        Display both the Plan B vesting curve and multiplier curve visualizations.

        Args:
            show_chart: Whether to display the chart (default: True)
        """
        from plotly.subplots import make_subplots

        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=('Vesting Curve', 'Stake Multiplier Curve'))

        # Generate data points
        days = np.linspace(0, self.max_lock_days, 1000)
        vesting_values = [self.calculate_multiplier(day) for day in days]
        multiplier_values = [self.calculate_multiplier(day) for day in days]

        # Add vesting curve
        fig.add_trace(go.Scatter(
            x=days,
            y=vesting_values,
            mode='lines',
            name='Vesting Curve',
            line=dict(color='green', width=3),
            hovertemplate='Days: %{x:.0f}<br>Vesting Value: %{y:.2f}<extra></extra>'
        ), row=1, col=1)
        # Add multiplier curve
        fig.add_trace(go.Scatter(
            x=days,
            y=multiplier_values,
            mode='lines',
            name='Stake Multiplier',
            line=dict(color='purple', width=3),
            hovertemplate='Days: %{x:.0f}<br>Multiplier: %{y:.2f}x<extra></extra>'
        ), row=2, col=1)

        # Update layout
        fig.update_layout(
            title={
                'text': 'Plan B: Vesting and Stake Multiplier Curves',
                'font': {'size': 24}
            },
            xaxis_title='Lock Duration (Days)',
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=1000,
        )
        if show_chart:
            fig.show()

        return fig

    def show_plot(self, years=2, show_chart=True):
        """
        Display the Plan B vesting curve visualization with exponential curve 2^n.

        Args:
            years: Number of years to display (default: 2)
            show_chart: Whether to display the chart (default: True)
        """
        import plotly.graph_objects as go
        import numpy as np

        # Generate data points - assuming 8 blocks per day
        blocks_per_day = 8
        days = np.linspace(0, years * 365, 1000)

        # Calculate exponential multiplier: 2^n where n is years
        # Convert days to years for the exponential calculation
        years_array = days / 365
        multipliers = np.power(2, years_array)

        # Create the figure
        fig = go.Figure()

        # Add the exponential multiplier curve
        fig.add_trace(go.Scatter(
            x=days,
            y=multipliers,
            mode='lines',
            name='Stake Multiplier (2^n)',
            line=dict(color='purple', width=3),
            hovertemplate='Days: %{x:.0f}<br>Years: %{customdata:.2f}<br>Multiplier: %{y:.2f}x<extra></extra>',
            customdata=years_array
        ))

        # Add milestone markers
        milestones = [
            (0, 'No lock'),
            (183, '6 months'),
            (365, '1 year'),
            (548, '18 months'),
            (730, '2 years')
        ]

        for days_locked, label in milestones:
            years_locked = days_locked / 365
            multiplier = np.power(2, years_locked)
            fig.add_trace(go.Scatter(
                x=[days_locked],
                y=[multiplier],
                mode='markers+text',
                name=label,
                marker=dict(size=12, color='red'),
                text=[f'{label}: {multiplier:.2f}x'],
                textposition='top center',
                showlegend=False
            ))

        # Update layout
        fig.update_layout(
            title={
                'text': 'Plan B: Exponential Stake-Time Multiplier Curve (2^n)',
                'font': {'size': 24}
            },
            xaxis_title='Lock Duration (Days)',
            yaxis_title='Multiplier (2^years)',
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=700,
            yaxis=dict(
                tickformat='.1f',
                gridcolor='lightgray',
                range=[0, max(5, np.power(2, years) * 1.1)],
                type='log'  # Use log scale for better visualization
            ),
            xaxis=dict(
                gridcolor='lightgray',
                range=[0, years * 365]
            ),
            annotations=[
                dict(
                    x=365,
                    y=2,
                    xref='x',
                    yref='y',
                    text='2x at 1 year',
                    showarrow=True,
                    arrowhead=2,
                    ax=50,
                    ay=-30
                ),
                dict(
                    x=730,
                    y=4,
                    xref='x',
                    yref='y',
                    text='4x at 2 years',
                    showarrow=True,
                    arrowhead=2,
                    ax=-50,
                    ay=-30
                ),
                dict(
                    x=days[-1] * 0.7,
                    y=multipliers[-1] * 0.7,
                    xref='x',
                    yref='y',
                    text=f'Exponential growth: 2^n<br>8 blocks/day runtime',
                    showarrow=False,
                    font=dict(size=14, color='purple'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='purple',
                    borderwidth=1
                )
            ]
        )

        # Add shaded regions for different tiers
        fig.add_shape(
            type='rect',
            x0=0, x1=365,
            y0=1, y1=2,
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(width=0),
            layer='below'
        )

        fig.add_shape(
            type='rect',
            x0=365, x1=730,
            y0=2, y1=4,
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(width=0),
            layer='below'
        )

        if show_chart:
            fig.show()

        return fig