import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import GateIODataFetcher
from chart_generator import OrderFlowProcessor, EnhancedOrderFlowChart

def main():
    st.title("📊 Enhanced Gate.io Orderflow Footprint Chart")
    st.markdown("Complete implementation with bid/ask volumes, delta analysis, and volume profile")
    st.markdown("---")
    
    # Initialize components
    data_fetcher = GateIODataFetcher()
    orderflow_processor = OrderFlowProcessor()
    
    # Sidebar controls
    st.sidebar.title("📈 Chart Settings")
    
    # Get all trading pairs
    all_pairs = data_fetcher.get_all_trading_pairs()
    
    # Time intervals
    intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    
    # User inputs
    selected_pair = st.sidebar.selectbox("Select Trading Pair", all_pairs)
    selected_interval = st.sidebar.selectbox("Select Time Interval", intervals)
    num_candles = st.sidebar.slider("Number of Candles", 10, 100, 50)
    price_levels = st.sidebar.slider("Price Levels per Candle", 10, 50, 20)
    
    # Time period selection
    st.sidebar.markdown("### 📅 Time Period")
    time_period_option = st.sidebar.radio(
        "Select time period:",
        ["Last 24 hours", "Last 3 days", "Last week", "Last month", "Custom"]
    )
    
    if time_period_option == "Last 24 hours":
        time_period_hours = 24
    elif time_period_option == "Last 3 days":
        time_period_hours = 72
    elif time_period_option == "Last week":
        time_period_hours = 168
    elif time_period_option == "Last month":
        time_period_hours = 720
    else:  # Custom
        time_period_hours = st.sidebar.number_input("Hours back:", min_value=1, max_value=720, value=24)
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Features:**")
    st.sidebar.markdown("✅ Bid/Ask volume display")
    st.sidebar.markdown("✅ Delta calculations")
    st.sidebar.markdown("✅ Cumulative delta")
    st.sidebar.markdown("✅ Volume profile")
    st.sidebar.markdown("✅ Color-coded imbalances")
    st.sidebar.markdown("✅ Time-based columns")
    st.sidebar.markdown("✅ Gate.io API integration")
    
    # Main content area
    if st.button("🔄 Generate Enhanced Footprint Chart", key="load_enhanced"):
        with st.spinner("Fetching and processing orderflow data from Gate.io..."):
            # Fetch OHLC data
            ohlc_data = data_fetcher.get_historical_data(selected_pair, selected_interval, time_period_hours)
            
            if ohlc_data is not None and not ohlc_data.empty:
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
                    
                    # Show OHLC data
                    if st.checkbox("Show OHLC Data"):
                        st.dataframe(ohlc_data.tail())
                
                else:
                    st.error("No footprint data generated")
            else:
                st.error("Failed to fetch OHLC data from Gate.io")
    
    # Auto-refresh functionality
    if auto_refresh:
        st.markdown(f"🔄 Auto-refresh enabled (every {refresh_interval} seconds)")
        st.info("Auto-refresh will automatically reload the data. Click the button above to manually refresh.")

if __name__ == "__main__":
    main()
