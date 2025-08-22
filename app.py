import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import plotly.express as px

# Binance API endpoints
BINANCE_API_BASE = "https://api.binance.com/api/v3"

class BinanceDataFetcher:
    def __init__(self):
        self.base_url = BINANCE_API_BASE
    
    def get_all_trading_pairs(self):
        """Get all available trading pairs from Binance"""
        endpoint = f"{self.base_url}/exchangeInfo"
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()
            
            trading_pairs = [symbol['symbol'] for symbol in data['symbols'] 
                            if symbol['status'] == 'TRADING' and symbol['quoteAsset'] == 'USDT']
            
            trading_pairs.sort()
            return trading_pairs
        except Exception as e:
            st.error(f"Error fetching trading pairs: {e}")
            return ['BTCUSDT', 'ETHUSDT']
    
    def get_klines(self, symbol, interval, limit=100):
        """Get OHLC data from Binance"""
        endpoint = f"{self.base_url}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                df[col] = df[col].astype(float)
            
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def get_agg_trades(self, symbol, start_time, end_time, limit=1000):
        """Get aggregated trades for a specific time period"""
        endpoint = f"{self.base_url}/aggTrades"
        params = {
            'symbol': symbol,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': limit
        }
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return None
                
            df = pd.DataFrame(data)
            df['T'] = pd.to_datetime(df['T'], unit='ms')  # Trade time
            df['p'] = df['p'].astype(float)  # Price
            df['q'] = df['q'].astype(float)  # Quantity
            df['m'] = df['m'].astype(bool)   # Was buyer the maker?
            
            return df
        except Exception as e:
            st.error(f"Error fetching trade data: {e}")
            return None

class OrderFlowProcessor:
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
                    (trades['p'] >= price_min) & 
                    (trades['p'] < price_max)
                ]
                
                if len(level_trades) == 0:
                    bid_vol, ask_vol = 0, 0
                else:
                    # Separate buy and sell trades
                    # m=True means buyer was maker (sell market order hit bid)
                    # m=False means seller was maker (buy market order hit ask)
                    sell_trades = level_trades[level_trades['m'] == True]  # Market sells
                    buy_trades = level_trades[level_trades['m'] == False]  # Market buys
                    
                    bid_vol = sell_trades['q'].sum()  # Volume hitting bids (sells)
                    ask_vol = buy_trades['q'].sum()   # Volume hitting asks (buys)
                
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
        fetcher = BinanceDataFetcher()
        return fetcher.get_agg_trades(symbol, start_time, end_time)
    
    def calculate_cumulative_delta(self, footprint_df):
        """Calculate cumulative delta for each candle"""
        candle_deltas = footprint_df.groupby('candle_idx')['delta'].sum().reset_index()
        candle_deltas['cumulative_delta'] = candle_deltas['delta'].cumsum()
        return candle_deltas

class EnhancedOrderFlowChart:
    def __init__(self, footprint_data, ohlc_data):
        self.footprint_data = footprint_data
        self.ohlc_data = ohlc_data
        self.fig = None
    
    def create_footprint_heatmap(self):
        """Create the main footprint heatmap chart"""
        # Pivot data for heatmap
        pivot_data = self.footprint_data.pivot_table(
            index='price', 
            columns='candle_idx', 
            values=['bid_volume', 'ask_volume', 'delta', 'total_volume'],
            fill_value=0
        )
        
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
                        opacity = min(abs(row['delta']) / candle_data['delta'].abs().max(), 1.0)
                    else:
                        color = 'red'
                        opacity = min(abs(row['delta']) / candle_data['delta'].abs().max(), 1.0)
                    
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

def main():
    st.title("📊 Enhanced Binance Orderflow Footprint Chart")
    st.markdown("Complete implementation with bid/ask volumes, delta analysis, and volume profile")
    st.markdown("---")
    
    # Initialize components
    data_fetcher = BinanceDataFetcher()
    orderflow_processor = OrderFlowProcessor()
    
    # Sidebar controls
    st.sidebar.title("📈 Chart Settings")
    
    all_pairs = data_fetcher.get_all_trading_pairs()
    intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    
    selected_pair = st.sidebar.selectbox("Select Trading Pair", all_pairs)
    selected_interval = st.sidebar.selectbox("Select Time Interval", intervals)
    num_candles = st.sidebar.slider("Number of Candles", 10, 50, 20)
    price_levels = st.sidebar.slider("Price Levels per Candle", 10, 50, 20)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Features:**")
    st.sidebar.markdown("✅ Bid/Ask volume display")
    st.sidebar.markdown("✅ Delta calculations")
    st.sidebar.markdown("✅ Cumulative delta")
    st.sidebar.markdown("✅ Volume profile")
    st.sidebar.markdown("✅ Color-coded imbalances")
    st.sidebar.markdown("✅ Time-based columns")
    
    if st.button("🔄 Generate Enhanced Footprint Chart", key="load_enhanced"):
        with st.spinner("Fetching and processing orderflow data..."):
            # Fetch OHLC data
            ohlc_data = data_fetcher.get_klines(selected_pair, selected_interval, num_candles)
            
            if ohlc_data is not None:
                # Create footprint matrix
                footprint_data = orderflow_processor.create_footprint_matrix(
                    ohlc_data, selected_pair, price_levels
                )
                
                if not footprint_data.empty:
                    # Create enhanced chart
                    chart = EnhancedOrderFlowChart(footprint_data, ohlc_data)
                    fig = chart.create_footprint_heatmap()
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display summary metrics
                    st.markdown("### 📊 Orderflow Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_delta = footprint_data['delta'].sum()
                    total_volume = footprint_data['total_volume'].sum()
                    
                    with col1:
                        st.metric("Current Price", f"${ohlc_data['close'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("Total Delta", f"{total_delta:.2f}")
                    with col3:
                        st.metric("Total Volume", f"{total_volume:.0f}")
                    with col4:
                        delta_color = "normal" if total_delta >= 0 else "inverse"
                        st.metric("Net Flow", "Bullish" if total_delta > 0 else "Bearish")
                    
                    # Show data tables
                    if st.checkbox("Show Footprint Data"):
                        st.dataframe(footprint_data.tail(20))
                
                else:
                    st.error("No footprint data generated")
            else:
                st.error("Failed to fetch OHLC data")

if __name__ == "__main__":
    main()
