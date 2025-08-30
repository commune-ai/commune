import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

class VestingCurve:
    def __init__(self, start_date=None, total_supply=1000000):
        self.start_date = start_date or datetime.now()
        self.total_supply = total_supply
    
    def linear_vesting(self, years, multiplier):
        """Create linear vesting curve over specified years with given multiplier"""
        days = years * 365
        time_points = np.linspace(0, days, 100)
        
        # Linear emission from 0 to total_supply * multiplier
        emissions = np.linspace(0, self.total_supply * multiplier, 100)
        
        dates = [self.start_date + timedelta(days=int(day)) for day in time_points]
        
        return dates, emissions
    
    def create_vesting_chart(self):
        """Create plotly chart showing both vesting curves"""
        # 1x at 1 year, 2x at 2 years curve
        dates_1, emissions_1 = self.linear_vesting(2, 2)
        
        # 4x at 2 years curve
        dates_2, emissions_2 = self.linear_vesting(2, 4)
        
        # Create figure
        fig = go.Figure()
        
        # Add 2x curve
        fig.add_trace(go.Scatter(
            x=dates_1,
            y=emissions_1,
            mode='lines',
            name='2x Vesting (2 years)',
            line=dict(color='blue', width=3),
            hovertemplate='Date: %{x}<br>Tokens: %{y:,.0f}<extra></extra>'
        ))
        
        # Add 4x curve
        fig.add_trace(go.Scatter(
            x=dates_2,
            y=emissions_2,
            mode='lines',
            name='4x Vesting (2 years)',
            line=dict(color='red', width=3),
            hovertemplate='Date: %{x}<br>Tokens: %{y:,.0f}<extra></extra>'
        ))
        
        # Add markers at 1 year and 2 years for 2x curve
        one_year_idx = 50  # Halfway point
        fig.add_trace(go.Scatter(
            x=[dates_1[one_year_idx]],
            y=[emissions_1[one_year_idx]],
            mode='markers+text',
            name='1 Year Mark',
            marker=dict(size=10, color='blue'),
            text=['1x (1 year)'],
            textposition='top center',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Linear Vesting Curves Comparison',
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
                gridcolor='lightgray'
            ),
            xaxis=dict(
                gridcolor='lightgray'
            )
        )
        
        # Add annotations
        fig.add_annotation(
            x=dates_1[-1],
            y=emissions_1[-1],
            text="2x at 2 years",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=-30
        )
        
        fig.add_annotation(
            x=dates_2[-1],
            y=emissions_2[-1],
            text="4x at 2 years",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=30
        )
        
        return fig
    
    def show_chart(self):
        """Display the vesting chart"""
        fig = self.create_vesting_chart()
        fig.show()
        return fig
    
    def save_chart(self, filename='vesting_curves.html'):
        """Save the chart to an HTML file"""
        fig = self.create_vesting_chart()
        fig.write_html(filename)
        return filename

# # Example usage
# if __name__ == "__main__":
#     vesting = VestingCurve(total_supply=1000000)
#     vesting.show_chart()
