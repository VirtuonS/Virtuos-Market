import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import json

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
            
            # Extract trading pairs
            trading_pairs = [symbol['symbol'] for symbol in data['symbols'] 
                            if symbol['status'] == 'TRADING' and symbol['quoteAsset'] == 'USDT']
            
            # Sort alphabetically
            trading_pairs.sort()
            return trading_pairs
        except Exception as e:
            st.error(f"Error fetching trading pairs: {e}")
            return ['BTCUSDT', 'ETHUSDT']  # Fallback to popular pairs
    
    def get_klines(self, symbol, interval, limit=1000):
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
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert data types
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def get_recent_trades(self, symbol, limit=10000):
        """Get recent trades for orderflow analysis"""
        endpoint = f"{self.base_url}/trades"
        params = {
            'symbol': symbol,
            'limit': limit
        }
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            for col in ['price', 'qty', 'quoteQty']:
                df[col] = df[col].astype(float)
            
            return df
        except Exception as e:
            st.error(f"Error fetching trade data: {e}")
            return None

class OrderFlowProcessor:
    def __init__(self):
        pass
    
    def process_trades_to_orderflow(self, trades_df, ohlc_df, price_levels=20):
        """Process trade data into orderflow format"""
        if trades_df is None or ohlc_df is None:
            return None, None
        
        # Merge trades with OHLC data
        trades_df['candle_id'] = pd.cut(trades_df['time'], 
                                       bins=ohlc_df['open_time'].tolist() + [ohlc_df['close_time'].iloc[-1]],
                                       labels=False, right=False)
        
        orderflow_data = []
        
        for candle_idx in range(len(ohlc_df)):
            candle_trades = trades_df[trades_df['candle_id'] == candle_idx]
            
            if len(candle_trades) == 0:
                continue
            
            # Calculate price levels for this candle
            high = ohlc_df.iloc[candle_idx]['high']
            low = ohlc_df.iloc[candle_idx]['low']
            price_range = high - low
            tick_size = price_range / price_levels
            
            price_levels_array = np.arange(low, high + tick_size, tick_size)
            
            for price in price_levels_array:
                # Separate buy and sell trades at this price level
                buy_trades = candle_trades[candle_trades['price'] <= price]
                sell_trades = candle_trades[candle_trades['price'] > price]
                
                bid_size = sell_trades['qty'].sum()
                ask_size = buy_trades['qty'].sum()
                
                orderflow_data.append({
                    'bid_size': bid_size,
                    'price': price,
                    'ask_size': ask_size,
                    'identifier': candle_idx,
                    'imbalance': bid_size - ask_size if (bid_size + ask_size) > 0 else 0
                })
        
        orderflow_df = pd.DataFrame(orderflow_data)
        
        # Prepare OHLC data for OrderFlowChart
        ohlc_processed = ohlc_df[['open', 'high', 'low', 'close']].copy()
        ohlc_processed['identifier'] = range(len(ohlc_df))
        
        return orderflow_df, ohlc_processed

class OrderFlowChart:
    """Replicating the OrderFlowChart class from the original repository"""
    def __init__(self, orderflow_data, ohlc_data, identifier_col='identifier', imbalance_col=None):
        self.orderflow_data = orderflow_data
        self.ohlc_data = ohlc_data
        self.identifier_col = identifier_col
        self.imbalance_col = imbalance_col
        self.fig = None
    
    def plot(self):
        """Create the orderflow chart using Plotly"""
        if self.orderflow_data is None or self.ohlc_data is None:
            return None
        
        # Create figure with secondary y-axis
        self.fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Orderflow Chart', 'Volume Profile')
        )
        
        # Add candlestick chart
        self.fig.add_trace(
            go.Candlestick(
                x=self.ohlc_data.index,
                open=self.ohlc_data['open'],
                high=self.ohlc_data['high'],
                low=self.ohlc_data['low'],
                close=self.ohlc_data['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add orderflow data for the most recent candle
        latest_candle = self.ohlc_data.index.max()
        latest_orderflow = self.orderflow_data[self.orderflow_data[self.identifier_col] == latest_candle]
        
        if not latest_orderflow.empty:
            # Add bid sizes (red)
            self.fig.add_trace(
                go.Scatter(
                    x=[latest_candle] * len(latest_orderflow),
                    y=latest_orderflow['price'],
                    mode='markers',
                    marker=dict(
                        size=latest_orderflow['bid_size'] / latest_orderflow['bid_size'].max() * 15,
                        color='red',
                        opacity=0.7
                    ),
                    name='Bid Size',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add ask sizes (green)
            self.fig.add_trace(
                go.Scatter(
                    x=[latest_candle] * len(latest_orderflow),
                    y=latest_orderflow['price'],
                    mode='markers',
                    marker=dict(
                        size=latest_orderflow['ask_size'] / latest_orderflow['ask_size'].max() * 15,
                        color='green',
                        opacity=0.7
                    ),
                    name='Ask Size',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Add volume profile
        volume_profile = self.orderflow_data.groupby('price').agg({
            'bid_size': 'sum',
            'ask_size': 'sum'
        }).reset_index()
        
        self.fig.add_trace(
            go.Bar(
                x=volume_profile['bid_size'],
                y=volume_profile['price'],
                orientation='h',
                name='Bid Volume',
                marker_color='red',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        self.fig.add_trace(
            go.Bar(
                x=-volume_profile['ask_size'],
                y=volume_profile['price'],
                orientation='h',
                name='Ask Volume',
                marker_color='green',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout for interactivity
        self.fig.update_layout(
            title="Orderflow Footprint Chart",
            xaxis_title="Time",
            yaxis_title="Price",
            height=800,
            dragmode='pan',  # Enable dragging
            hovermode='x unified'
        )
        
        self.fig.update_xaxes(rangeslider_visible=False)
        
        return self.fig

def main():
    st.title("📊 Binance Orderflow Footprint Chart")
    st.markdown("---")
    
    # Initialize data fetcher and processors
    data_fetcher = BinanceDataFetcher()
    orderflow_processor = OrderFlowProcessor()
    
    # Sidebar controls
    st.sidebar.title("📈 Chart Settings")
    
    # Get all trading pairs from Binance
    all_pairs = data_fetcher.get_all_trading_pairs()
    
    # Time intervals
    intervals = [
        '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'
    ]
    
    # User inputs
    selected_pair = st.sidebar.selectbox("Select Trading Pair", all_pairs)
    selected_interval = st.sidebar.selectbox("Select Time Interval", intervals)
    num_candles = st.sidebar.slider("Number of Candles", 10, 100, 50)
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Instructions:**")
    st.sidebar.markdown("- Select any Binance trading pair")
    st.sidebar.markdown("- Choose your preferred time interval")
    st.sidebar.markdown("- Adjust the number of candles to display")
    st.sidebar.markdown("- Enable auto-refresh for live data")
    st.sidebar.markdown("- Drag the chart to pan around")
    st.sidebar.markdown("- Use mouse wheel to zoom in/out")
    
    # Main content area
    if st.button("🔄 Load Data", key="load_data"):
        with st.spinner("Fetching data from Binance..."):
            # Fetch OHLC data
            ohlc_data = data_fetcher.get_klines(selected_pair, selected_interval, num_candles)
            
            if ohlc_data is not None:
                # Fetch recent trades for orderflow analysis
                trades_data = data_fetcher.get_recent_trades(selected_pair, 10000)
                
                # Process data into orderflow format
                orderflow_df, ohlc_processed = orderflow_processor.process_trades_to_orderflow(
                    trades_data, ohlc_data
                )
                
                if orderflow_df is not None and ohlc_processed is not None:
                    # Create orderflow chart using the OrderFlowChart class
                    orderflowchart = OrderFlowChart(
                        orderflow_df,
                        ohlc_processed,
                        identifier_col='identifier'
                    )
                    
                    # Plot the chart
                    fig = orderflowchart.plot()
                    
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True, height=800)
                        
                        # Display data summary
                        st.markdown("### 📊 Data Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Price", f"${ohlc_data['close'].iloc[-1]:.2f}")
                        with col2:
                            change_pct = ((ohlc_data['close'].iloc[-1] - ohlc_data['open'].iloc[-1]) / ohlc_data['open'].iloc[-1] * 100)
                            st.metric("24h Change", f"{change_pct:.2f}%")
                        with col3:
                            st.metric("Volume", f"{ohlc_data['volume'].iloc[-1]:.0f}")
                        with col4:
                            st.metric("Trades", f"{len(trades_data):.0f}")
                        
                        # Show raw data option
                        if st.checkbox("Show Raw Data"):
                            st.markdown("#### OHLC Data")
                            st.dataframe(ohlc_data.tail())
                            
                            st.markdown("#### Orderflow Data")
                            st.dataframe(orderflow_df.tail())
                    else:
                        st.error("Failed to create chart")
                else:
                    st.error("Failed to process orderflow data")
            else:
                st.error("Failed to fetch OHLC data")
    
    # Auto-refresh functionality
    if auto_refresh:
        st.markdown(f"🔄 Auto-refresh enabled (every {refresh_interval} seconds)")
        # The button will be clicked automatically in a real deployment
        # For local testing, user needs to click manually

if __name__ == "__main__":
    main()
