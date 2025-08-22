import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

class GateIODataFetcher:
    """Handles all API connections and data fetching from Gate.io"""
    
    def __init__(self):
        self.base_url = "https://api.gate.io/api/v4"
        # Fallback list of popular trading pairs
        self.fallback_pairs = [
            "BTC_USDT", "ETH_USDT", "BNB_USDT", "ADA_USDT", "DOT_USDT", 
            "SOL_USDT", "MATIC_USDT", "AVAX_USDT", "LINK_USDT", "UNI_USDT",
            "XRP_USDT", "LTC_USDT", "BCH_USDT", "ALGO_USDT", "VET_USDT"
        ]
    
    def get_all_trading_pairs(self):
        """Get all available trading pairs from Gate.io"""
        try:
            endpoint = f"{self.base_url}/spot/currency_pairs"
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract USDT trading pairs
            trading_pairs = [pair['id'] for pair in data if pair['id'].endswith('_USDT')]
            trading_pairs.sort()
            return trading_pairs
        except Exception as e:
            print(f"Error fetching trading pairs: {e}")
            return self.fallback_pairs
    
    def get_klines(self, symbol, interval, limit=1000):
        """Get OHLC data from Gate.io"""
        try:
            # Map intervals to Gate.io format
            interval_map = {
                '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h',
                '12h': '12h', '1d': '1d'
            }
            
            gate_interval = interval_map.get(interval, '1h')
            endpoint = f"{self.base_url}/spot/candlesticks"
            params = {
                'currency_pair': symbol,
                'interval': gate_interval,
                'limit': limit
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'volume', 'close', 'high', 'low', 'open', 'ignore'
            ])
            
            # Convert data types
            df['open_time'] = pd.to_datetime(df['timestamp'], unit='s')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Add close_time (next candle's open_time)
            df['close_time'] = df['open_time'].shift(-1)
            df.loc[df.index[-1], 'close_time'] = df['open_time'].iloc[-1] + pd.Timedelta(gate_interval)
            
            # Reorder columns to match expected format
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
            
            return df.sort_values('open_time').reset_index(drop=True)
            
        except Exception as e:
            print(f"Error fetching klines: {e}")
            return None
    
    def get_trades(self, symbol, start_time=None, end_time=None, limit=1000):
        """Get trade data from Gate.io"""
        try:
            endpoint = f"{self.base_url}/spot/trades"
            params = {
                'currency_pair': symbol,
                'limit': limit
            }
            
            if start_time:
                params['from'] = int(start_time.timestamp())
            if end_time:
                params['to'] = int(end_time.timestamp())
            
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['create_time_ms'], unit='ms')
            df['price'] = df['price'].astype(float)
            df['qty'] = df['amount'].astype(float)
            df['m'] = df['side'] == 'sell'  # True for sell orders (maker)
            
            return df[['time', 'price', 'qty', 'm']]
            
        except Exception as e:
            print(f"Error fetching trades: {e}")
            return None
    
    def get_historical_data(self, symbol, interval, time_period_hours=24):
        """Get historical data for specified time period"""
        try:
            # Calculate number of candles needed
            interval_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
                '12h': 720, '1d': 1440
            }
            
            minutes_per_candle = interval_minutes.get(interval, 60)
            total_candles = int((time_period_hours * 60) / minutes_per_candle)
            
            # Limit to prevent API overload
            total_candles = min(total_candles, 1000)
            
            return self.get_klines(symbol, interval, total_candles)
            
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return None
    
    def get_sample_data(self, symbol="BTC_USDT", limit=50):
        """Generate sample data when API is not available"""
        np.random.seed(42)
        
        # Determine base price based on symbol
        if 'BTC' in symbol:
            base_price = 30000
        elif 'ETH' in symbol:
            base_price = 2000
        elif 'BNB' in symbol:
            base_price = 300
        else:
            base_price = 1
        
        # Generate random OHLC data
        dates = pd.date_range(start='2023-01-01', periods=limit, freq='H')
        
        ohlc_df = pd.DataFrame({
            'open_time': dates,
            'open': [base_price + np.random.normal(0, base_price * 0.01) for _ in range(limit)],
            'high': [base_price + abs(np.random.normal(base_price * 0.02, base_price * 0.005)) for _ in range(limit)],
            'low': [base_price - abs(np.random.normal(base_price * 0.02, base_price * 0.005)) for _ in range(limit)],
            'close': [base_price + np.random.normal(0, base_price * 0.01) for _ in range(limit)],
            'volume': [np.random.uniform(100, 1000) for _ in range(limit)],
            'close_time': dates + pd.Timedelta(hours=1)
        })
        
        # Generate random trades data
        trades_data = []
        for time in dates:
            num_trades = np.random.randint(10, 50)
            for _ in range(num_trades):
                price = base_price + np.random.normal(0, base_price * 0.01)
                qty = np.random.uniform(0.1, 5)
                trades_data.append({
                    'time': time + pd.Timedelta(minutes=np.random.randint(0, 60)),
                    'price': price,
                    'qty': qty,
                    'm': np.random.choice([True, False])
                })
        
        trades_df = pd.DataFrame(trades_data)
        
        return ohlc_df, trades_df
