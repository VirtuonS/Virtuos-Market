import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class OrderFlowProcessor:
    """Processes orderflow data and creates footprint matrices"""
    
    def __init__(self, tick_size=0.01):
        self.tick_size = tick_size
    
    def create_footprint_matrix(self, ohlc_data, symbol, num_levels=20):
        """Create a complete footprint matrix with bid/ask volumes"""
        footprint_data = []
        
        for idx, candle in ohlc_data.iterrows():
            # Get trades for this candle period
            trades = self.get_candle_trades(symbol, candle['open_time'], candle['close_time'])
            
            if trades is None or len(trades) == 0:
                # Create empty footprint for this candle
                price_levels = np.linspace(candle['low'], candle['high'], num_levels)
                for price in price_levels:
                    footprint_data.append({
                        'candle_idx': idx,
                        'time': candle['open_time'],
                        'price': round(price, 2),
                        'bid_volume': 0,
                        'ask_volume': 0,
                        'delta': 0,
                        'total_volume': 0
                    })
                continue
            
            # Process trades into price levels
            price_levels = np.linspace(candle['low'], candle['high'], num_levels)
            
            for i, price_level in enumerate(price_levels):
                # Define price range for this level
                if i == 0:
                    price_min = candle['low']
                else:
                    price_min = price_levels[i-1]
                
                if i == len(price_levels) - 1:
                    price_max = candle['high']
                else:
                    price_max = price_levels[i+1]
                
                # Filter trades in this price range
                level_trades = trades[
                    (trades['price'] >= price_min) & 
                    (trades['price'] < price_max)
                ]
                
                if len(level_trades) == 0:
                    bid_vol, ask_vol = 0, 0
                else:
                    # Separate buy and sell trades
                    # m=True means seller was maker (sell market order hit bid)
                    # m=False means buyer was maker (buy market order hit ask)
                    sell_trades = level_trades[level_trades['m'] == True]  # Market sells
                    buy_trades = level_trades[level_trades['m'] == False]  # Market buys
                    
                    bid_vol = sell_trades['qty'].sum()  # Volume hitting bids (sells)
                    ask_vol = buy_trades['qty'].sum()   # Volume hitting asks (buys)
                
                delta = ask_vol - bid_vol
                total_vol = bid_vol + ask_vol
                
                footprint_data.append({
                    'candle_idx': idx,
                    'time': candle['open_time'],
                    'price': round(price_level, 2),
                    'bid_volume': round(bid_vol, 2),
                    'ask_volume': round(ask_vol, 2),
                    'delta': round(delta, 2),
                    'total_volume': round(total_vol, 2)
                })
        
        return pd.DataFrame(footprint_data)
    
    def get_candle_trades(self, symbol, start_time, end_time):
        """Get trades for a specific candle period"""
        from data_fetcher import GateIODataFetcher
        fetcher = GateIODataFetcher()
        return fetcher.get_trades(symbol, start_time, end_time)
    
    def calculate_cumulative_delta(self, footprint_df):
        """Calculate cumulative delta for each candle"""
        candle_deltas = footprint_df.groupby('candle_idx')['delta'].sum().reset_index()
        candle_deltas['cumulative_delta'] = candle_deltas['delta'].cumsum()
        return candle_deltas

class EnhancedOrderFlowChart:
    """Creates enhanced orderflow charts with advanced visualization"""
    
    def __init__(self, footprint_data, ohlc_data):
        self.footprint_data = footprint_data
        self.ohlc_data = ohlc_data
        self.fig = None
    
    def create_footprint_heatmap(self):
        """Create the main footprint heatmap chart"""
        # Create subplots
        self.fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.2, 0.1],
            subplot_titles=['Orderflow Footprint', 'Volume Profile', 'Delta & Cumulative Delta']
        )
        
        # Main footprint visualization
        self._add_footprint_cells()
        
        # Add candlestick overlay
        self._add_candlestick_overlay()
        
        # Add volume profile
        self._add_volume_profile()
        
        # Add delta metrics
        self._add_delta_metrics()
        
        # Update layout
        self.fig.update_layout(
            title="Enhanced Orderflow Footprint Chart",
            height=1000,
            showlegend=True,
            xaxis_title="Time",
            yaxis_title="Price"
        )
        
        return self.fig
    
    def _add_footprint_cells(self):
        """Add individual footprint cells with bid/ask volumes"""
        # Group by price and candle for cell creation
        for candle_idx in self.footprint_data['candle_idx'].unique():
            candle_data = self.footprint_data[self.footprint_data['candle_idx'] == candle_idx]
            
            for _, row in candle_data.iterrows():
                if row['total_volume'] > 0:
                    # Determine cell color based on delta
                    if row['delta'] > 0:
                        color = 'green'
                        opacity = min(abs(row['delta']) / candle_data['delta'].abs().max(), 1.0) if candle_data['delta'].abs().max() > 0 else 0.5
                    else:
                        color = 'red'
                        opacity = min(abs(row['delta']) / candle_data['delta'].abs().max(), 1.0) if candle_data['delta'].abs().max() > 0 else 0.5
                    
                    # Add cell as a rectangle with annotations
                    self.fig.add_shape(
                        type="rect",
                        x0=candle_idx-0.4, x1=candle_idx+0.4,
                        y0=row['price']-0.1, y1=row['price']+0.1,
                        fillcolor=color,
                        opacity=opacity * 0.5,
                        line=dict(width=1, color="white"),
                        row=1, col=1
                    )
                    
                    # Add bid volume text (left side)
                    if row['bid_volume'] > 0:
                        self.fig.add_annotation(
                            x=candle_idx-0.2, y=row['price'],
                            text=str(int(row['bid_volume'])),
                            showarrow=False,
                            font=dict(size=8, color="red"),
                            row=1, col=1
                        )
                    
                    # Add ask volume text (right side)
                    if row['ask_volume'] > 0:
                        self.fig.add_annotation(
                            x=candle_idx+0.2, y=row['price'],
                            text=str(int(row['ask_volume'])),
                            showarrow=False,
                            font=dict(size=8, color="green"),
                            row=1, col=1
                        )
    
    def _add_candlestick_overlay(self):
        """Add candlestick overlay on the footprint"""
        self.fig.add_trace(
            go.Candlestick(
                x=self.ohlc_data.index,
                open=self.ohlc_data['open'],
                high=self.ohlc_data['high'],
                low=self.ohlc_data['low'],
                close=self.ohlc_data['close'],
                name="OHLC",
                opacity=0.7
            ),
            row=1, col=1
        )
    
    def _add_volume_profile(self):
        """Add horizontal volume profile"""
        volume_profile = self.footprint_data.groupby('price').agg({
            'bid_volume': 'sum',
            'ask_volume': 'sum',
            'total_volume': 'sum'
        }).reset_index()
        
        # Bid volume (red, negative direction)
        self.fig.add_trace(
            go.Bar(
                x=-volume_profile['bid_volume'],
                y=volume_profile['price'],
                orientation='h',
                name='Bid Volume',
                marker_color='red',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Ask volume (green, positive direction)
        self.fig.add_trace(
            go.Bar(
                x=volume_profile['ask_volume'],
                y=volume_profile['price'],
                orientation='h',
                name='Ask Volume',
                marker_color='green',
                opacity=0.7
            ),
            row=2, col=1
        )
    
    def _add_delta_metrics(self):
        """Add delta and cumulative delta"""
        processor = OrderFlowProcessor()
        delta_data = processor.calculate_cumulative_delta(self.footprint_data)
        
        # Delta bar chart
        self.fig.add_trace(
            go.Bar(
                x=delta_data['candle_idx'],
                y=delta_data['delta'],
                name='Delta',
                marker_color=['green' if d > 0 else 'red' for d in delta_data['delta']]
            ),
            row=3, col=1
        )
        
        # Cumulative delta line
        self.fig.add_trace(
            go.Scatter(
                x=delta_data['candle_idx'],
                y=delta_data['cumulative_delta'],
                mode='lines',
                name='Cumulative Delta',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
    
    def show(self):
        """Display the chart"""
        if self.fig is not None:
            self.fig.show()
        else:
            print("No chart to display. Create a chart first.")
