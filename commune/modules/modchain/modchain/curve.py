import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import plotly.io as pio

class Curve:
    def __init__(self,
                start_date = datetime.now(),  # Bitcoin genesis block
                initial_supply = 0,
                max_supply = 42_000_000,
                initial_block_reward = 100,
                halving_interval = 210_000  ,# blocks
                blocks_per_day = 144,  # approximately 10 minutes per bloc
    ):
        kwargs = locals()
        kwargs.pop("self")
        self.__dict__.update(kwargs)

    def bitcoin_supply_at_block(self, block_height):
        """Calculate Bitcoin supply at a given block height with halvings"""
        supply = 0
        current_reward = self.initial_block_reward
        blocks_processed = 0
        
        while blocks_processed < block_height:
            blocks_in_period = min(self.halving_interval, block_height - blocks_processed)
            supply += blocks_in_period * current_reward
            blocks_processed += blocks_in_period
            current_reward /= 2
            
        return min(supply, self.max_supply)
    
    def exponential_approximation(self, days):
        """Create exponential curve that approximates Bitcoin supply without halvings"""
        # Parameters fitted to match Bitcoin's supply curve
        # Using formula: supply = max_supply * (1 - e^(-k*t))
        k = 0.00035  # decay constant adjusted to match Bitcoin's emission
        supply = self.max_supply * (1 - np.exp(-k * days))
        return supply
    
    def create_plotly_chart(self, years=50):
        """Create Plotly chart comparing Bitcoin supply curve with exponential approximation"""
        days = years * 365
        time_points = np.linspace(0, days, 1000)
        
        # Calculate Bitcoin supply with halvings
        bitcoin_supply = []
        dates = []
        for day in time_points:
            block_height = int(day * self.blocks_per_day)
            supply = self.bitcoin_supply_at_block(block_height)
            bitcoin_supply.append(supply)
            dates.append(self.start_date + timedelta(days=int(day)))
        
        # Calculate exponential approximation
        exponential_supply = [self.exponential_approximation(day) for day in time_points]
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add Bitcoin supply curve (with halvings)
        fig.add_trace(go.Scatter(
            x=dates,
            y=bitcoin_supply,
            mode='lines',
            name='Bitcoin Supply (with halvings)',
            line=dict(color='orange', width=3),
            hovertemplate='Date: %{x}<br>Supply: %{y:,.0f} BTC<extra></extra>'
        ))
        
        # Add exponential approximation
        fig.add_trace(go.Scatter(
            x=dates,
            y=exponential_supply,
            mode='lines',
            name='Exponential Approximation (no halvings)',
            line=dict(color='blue', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>Supply: %{y:,.0f} BTC<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Bitcoin Supply Curve vs Exponential Approximation',
                'font': {'size': 24}
            },
            xaxis_title='Date',
            yaxis_title='Total Supply (BTC)',
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
        
        # Add horizontal line at max supply
        fig.add_hline(
            y=self.max_supply, 
            line_dash="dot", 
            line_color="red",
            annotation_text="Max Supply: 21M BTC",
            annotation_position="right"
        )
        
        return fig
    
    def save_chart(self, filename='bitcoin_curve.html', years=50):
        """Save the chart to an HTML file"""
        fig = self.create_plotly_chart(years)
        pio.write_html(fig, filename)
        return filename
    
    def show_chart(self, years=50):
        """Display the chart in browser"""
        fig = self.create_plotly_chart(years)
        fig.show()
        return fig


    def bitcoin_curve(self, years=50, filename=None):
        return Curve(max_supply=21_000_000, initial_block_reward=50, start_date=datetime(2009, 1, 3), ).show_chart(years=years)
    def modchain_curve(self, years=50, filename=None):
        return Curve(max_supply=42_000_000, initial_block_reward=100, start_date=datetime(2023, 1, 1), ).show_chart(years=years)

    def both_curves(self, years=50, filename=None):
        fig = self.create_plotly_chart(years)
        bitcoin_curve = Curve(max_supply=21_000_000, initial_block_reward=50, start_date=datetime(2009, 1, 3), )
        modchain_curve = Curve(max_supply=42_000_000, initial_block_reward=100, start_date=datetime(2023, 1, 1), )
        
        fig.add_trace(bitcoin_curve.create_plotly_chart(years).data[0])
        fig.add_trace(modchain_curve.create_plotly_chart(years).data[0])
        
        fig.show()
        if filename:
            pio.write_html(fig, filename)
        
        return fig


    

    def curve3(self, years=50, filename=None):
        """Create exponential approximation curve without halvings"""
        fig = go.Figure()

        days = years * 365
        time_points = np.linspace(0, days, 1000)
        dates = [self.start_date + timedelta(days=int(day)) for day in time_points]

        # Calculate exponential approximation
        exponential_supply = [self.exponential_approximation(day) for day in time_points]

        # Add exponential approximation
        fig.add_trace(go.Scatter(
            x=dates,
            y=exponential_supply,
            mode='lines',
            name='Exponential Approximation (no halvings)',
            line=dict(color='green', width=3),
            hovertemplate='Date: %{x}<br>Supply: %{y:,.0f} BTC<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title={
                'text': 'Exponential Approximation Curve',
                'font': {'size': 24}
            },
            xaxis_title='Date',
            yaxis_title='Total Supply',
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=700,
            yaxis=dict(
                tickformat=',.0f',
                gridcolor='lightgray'
            ),
            xaxis=dict(
                gridcolor='lightgray'
            )
        )

        # Add horizontal line at max supply
        fig.add_hline(
            y=self.max_supply, 
            line_dash="dot", 
            line_color="red",
            annotation_text=f"Max Supply: {self.max_supply:,.0f}",
            annotation_position="right"
        )

        fig.show()
        if filename:
            pio.write_html(fig, filename)

        return fig

    def show_plot(self, years=50, filename=None):
        """Create exponential curve representing 2^n where n is years, assuming 8 block runtime"""
        import plotly.graph_objects as go
        import numpy as np
        from datetime import datetime, timedelta
        import plotly.io as pio

        # Create time points for n years
        days = years * 365
        time_points = np.linspace(0, days, 1000)
        dates = [self.start_date + timedelta(days=int(day)) for day in time_points]

        # Calculate 2^n where n is years (convert days to years)
        years_array = time_points / 365
        exponential_values = 2 ** years_array

        # Scale to reasonable token values (multiply by base amount)
        base_tokens = 1_000_000  # Base amount of tokens
        token_values = exponential_values * base_tokens

        # Create figure
        fig = go.Figure()

        # Add exponential curve
        fig.add_trace(go.Scatter(
            x=dates,
            y=token_values,
            mode='lines',
            name='Exponential Growth (2^n)',
            line=dict(color='purple', width=3),
            hovertemplate='Date: %{x}<br>Tokens: %{y:,.0f}<br>Years: %{customdata:.2f}<extra></extra>',
            customdata=years_array
        ))

        # Add markers at key years
        key_years = [1, 2, 5, 10, 20, 30, 40, 50]
        for year in key_years:
            if year <= years:
                idx = int((year * 365 / days) * len(time_points))
                if idx < len(dates):
                    fig.add_trace(go.Scatter(
                        x=[dates[idx]],
                        y=[token_values[idx]],
                        mode='markers+text',
                        marker=dict(size=8, color='red'),
                        text=[f'{year}y: {2**year:.0f}x'],
                        textposition='top center',
                        showlegend=False
                    ))

        # Update layout
        fig.update_layout(
            title={
                'text': f'Exponential Token Growth: 2^n (n=years, 8 block runtime)',
                'font': {'size': 24}
            },
            xaxis_title='Date',
            yaxis_title='Total Tokens',
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
                type='log',  # Use log scale for exponential growth
                tickformat='.0e',
                gridcolor='lightgray'
            ),
            xaxis=dict(
                gridcolor='lightgray'
            )
        )

        # Add annotation explaining the curve
        fig.add_annotation(
            x=0.5,
            y=0.95,
            xref='paper',
            yref='paper',
            text='Growth Rate: 2^n where n = years<br>Assumes 8 block runtime',
            showarrow=False,
            font=dict(size=14),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )

        fig.show()
        if filename:
            pio.write_html(fig, filename)

        return fig

    def show_plot(self, years=50, filename=None):
        """Create exponential curve representing 2^n where n is years, assuming 8 block runtime"""
        import plotly.graph_objects as go
        import numpy as np
        from datetime import datetime, timedelta
        import plotly.io as pio

        # Create time points for n years
        days = years * 365
        time_points = np.linspace(0, days, 1000)
        dates = [self.start_date + timedelta(days=int(day)) for day in time_points]

        # Calculate 2^n where n is years (convert days to years)
        years_array = time_points / 365
        exponential_values = 2 ** years_array

        # Scale to reasonable token values (multiply by base amount)
        base_tokens = 1_000_000  # Base amount of tokens
        token_values = exponential_values * base_tokens

        # Create figure
        fig = go.Figure()

        # Add exponential curve
        fig.add_trace(go.Scatter(
            x=dates,
            y=token_values,
            mode='lines',
            name='Exponential Growth (2^n)',
            line=dict(color='purple', width=3),
            hovertemplate='Date: %{x}<br>Tokens: %{y:,.0f}<br>Years: %{customdata:.2f}<extra></extra>',
            customdata=years_array
        ))

        # Add markers at key years
        key_years = [1, 2, 5, 10, 20, 30, 40, 50]
        for year in key_years:
            if year <= years:
                idx = int((year * 365 / days) * len(time_points))
                if idx < len(dates):
                    fig.add_trace(go.Scatter(
                        x=[dates[idx]],
                        y=[token_values[idx]],
                        mode='markers+text',
                        marker=dict(size=8, color='red'),
                        text=[f'{year}y: {2**year:.0f}x'],
                        textposition='top center',
                        showlegend=False
                    ))

        # Update layout
        fig.update_layout(
            title={
                'text': f'Exponential Token Growth: 2^n (n=years, 8 block runtime)',
                'font': {'size': 24}
            },
            xaxis_title='Date',
            yaxis_title='Total Tokens',
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
                type='log',  # Use log scale for exponential growth
                tickformat='.0e',
                gridcolor='lightgray'
            ),
            xaxis=dict(
                gridcolor='lightgray'
            )
        )

        # Add annotation explaining the curve
        fig.add_annotation(
            x=0.5,
            y=0.95,
            xref='paper',
            yref='paper',
            text='Growth Rate: 2^n where n = years<br>Assumes 8 block runtime',
            showarrow=False,
            font=dict(size=14),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )

        fig.show()
        if filename:
            pio.write_html(fig, filename)

        return fig