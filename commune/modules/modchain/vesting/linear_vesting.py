import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

class LinearVesting:
    def __init__(self, start_date=None, total_supply=4_200_000):
        self.start_date = start_date or datetime.now()
        self.total_supply = total_supply
    
    def create_vesting_chart(self):
        """Create plotly chart showing both linear vesting curves"""
        # Time points from 0 to 2 years
        days = 2 * 365
        time_points = np.linspace(0, days, 1000)
        dates = [self.start_date + timedelta(days=int(day)) for day in time_points]
        
        # Curve 1: Piecewise linear - 0 to 1x at 1 year, 1x to 2x at 2 years
        curve1_values = []
        for day in time_points:
            if day <= 365:  # First year: 0 to 1x
                value = (day / 365) * self.total_supply
            else:  # Second year: 1x to 2x
                value = self.total_supply + ((day - 365) / 365) * self.total_supply
            curve1_values.append(value)
        
        # Curve 2: Piecewise linear - 0 to 1x at 1 year, 1x to 4x at 2 years
        curve2_values = []
        for day in time_points:
            if day <= 365:  # First year: 0 to 1x (same as curve 1)
                value = (day / 365) * self.total_supply
            else:  # Second year: 1x to 4x
                value = self.total_supply + ((day - 365) / 365) * 3 * self.total_supply
            curve2_values.append(value)
        
        # Create figure
        fig = go.Figure()
        
        # Add Curve 1 (2x at 2 years)
        fig.add_trace(go.Scatter(
            x=dates,
            y=curve1_values,
            mode='lines',
            name='Curve 1: 2x at 2 years',
            line=dict(color='blue', width=3),
            hovertemplate='Date: %{x}<br>Tokens Vested: %{y:,.0f}<extra></extra>'
        ))
        
        # Add Curve 2 (4x at 2 years)
        fig.add_trace(go.Scatter(
            x=dates,
            y=curve2_values,
            mode='lines',
            name='Curve 2: 4x at 2 years',
            line=dict(color='red', width=3),
            hovertemplate='Date: %{x}<br>Tokens Vested: %{y:,.0f}<extra></extra>'
        ))
        
        # Add markers at key points
        # 1 year mark (both curves at 1x)
        one_year_date = self.start_date + timedelta(days=365)
        fig.add_trace(go.Scatter(
            x=[one_year_date],
            y=[self.total_supply],
            mode='markers+text',
            name='1 Year Mark',
            marker=dict(size=12, color='green'),
            text=['1x (1 year)'],
            textposition='top center',
            showlegend=False
        ))
        
        # 2 year marks
        two_year_date = self.start_date + timedelta(days=730)
        fig.add_trace(go.Scatter(
            x=[two_year_date, two_year_date],
            y=[2 * self.total_supply, 4 * self.total_supply],
            mode='markers+text',
            name='2 Year Marks',
            marker=dict(size=12, color='purple'),
            text=['2x', '4x'],
            textposition='top center',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Linear Vesting Curves: Piecewise Linear Emission',
                'font': {'size': 24}
            },
            xaxis_title='Date',
            yaxis_title='Total Tokens Vested',
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=700,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            yaxis=dict(
                tickformat=',.0f',
                gridcolor='lightgray',
                range=[0, 4.5 * self.total_supply]
            ),
            xaxis=dict(
                gridcolor='lightgray'
            )
        )
        
        # Add annotations
        fig.add_annotation(
            x=two_year_date,
            y=2 * self.total_supply,
            text="Curve 1: 2x",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=-30
        )
        
        fig.add_annotation(
            x=two_year_date,
            y=4 * self.total_supply,
            text="Curve 2: 4x",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=30
        )
        
        # Add annotation for intersection point
        fig.add_annotation(
            x=one_year_date,
            y=self.total_supply,
            text="Curves intersect at (1 year, 1x)",
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=-50
        )
        
        return fig
    
    def show_chart(self):
        """Display the vesting chart"""
        fig = self.create_vesting_chart()
        fig.show()
        return fig
    
    def save_chart(self, filename='linear_vesting_curves.html'):
        """Save the chart to an HTML file"""
        fig = self.create_vesting_chart()
        fig.write_html(filename)
        return filename


    def exponential_approximation(self, days):
        """Create exponential curve that approximates curve 2 (4x at 2 years)"""
        # Using formula: supply = max_supply * multiplier * (1 - e^(-k*t))
        # k chosen so that at 2 years (730 days) we reach 4x
        # 4 = 4 * (1 - e^(-k*730))
        # 1 = 1 - e^(-k*730)
        # e^(-k*730) = 0
        # For practical purposes, we want e^(-k*730) ≈ 0.01
        # -k*730 = ln(0.01) = -4.605
        # k = 4.605/730 ≈ 0.00631
        k = 0.00631

        # Calculate exponential curve
        curve_values = []
        for day in days:
            if day <= 365:  # First year matches linear curve
                value = (day / 365) * self.total_supply
            else:  # After first year, use exponential
                # Start exponential from 1x at year 1
                days_after_year1 = day - 365
                # Exponential growth from 1x to 4x over second year
                exponential_multiplier = 1 + 3 * (1 - np.exp(-k * days_after_year1))
                value = self.total_supply * exponential_multiplier
            curve_values.append(value)

        return curve_values
# # Example usage
# if __name__ == "__main__":
#     vesting = LinearVesting(total_supply=4_200_000)
#     vesting.show_chart()
