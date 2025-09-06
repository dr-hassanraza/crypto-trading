import asyncio
import websocket
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import threading
import time
import logging
from collections import defaultdict, deque

from binance import ThreadedWebSocketManager
from config.config import Config

class RealTimeDataStreamer:
    def __init__(self):
        self.config = Config()
        self.is_running = False
        self.callbacks = defaultdict(list)  # event_type -> [callback_functions]
        self.price_data = defaultdict(deque)  # symbol -> price history
        self.volume_data = defaultdict(deque)  # symbol -> volume history
        self.order_book_data = {}  # symbol -> order book
        self.trade_data = defaultdict(deque)  # symbol -> recent trades
        
        # WebSocket managers
        self.binance_manager = None
        self.websocket_connections = {}
        
        # Data aggregation
        self.candle_data = defaultdict(lambda: defaultdict(list))  # symbol -> timeframe -> candles
        self.last_update = {}
        
        # Performance tracking
        self.message_count = 0
        self.last_message_time = datetime.now()
        
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback function for specific event types."""
        self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """Remove callback function."""
        if callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    def start_streaming(self, symbols: List[str]):
        """Start real-time data streaming for specified symbols."""
        if self.is_running:
            logging.warning("Streaming is already running")
            return
        
        self.is_running = True
        logging.info(f"Starting real-time streaming for symbols: {symbols}")
        
        # Start Binance streams if API keys available
        if self.config.BINANCE_API_KEY and self.config.BINANCE_SECRET_KEY:
            self._start_binance_streams(symbols)
        
        # Start public WebSocket streams
        self._start_public_streams(symbols)
        
        # Start data processing thread
        threading.Thread(target=self._data_processor, daemon=True).start()
        
        logging.info("Real-time streaming started successfully")
    
    def stop_streaming(self):
        """Stop all streaming connections."""
        self.is_running = False
        
        if self.binance_manager:
            self.binance_manager.stop()
        
        for ws in self.websocket_connections.values():
            try:
                ws.close()
            except:
                pass
        
        logging.info("Real-time streaming stopped")
    
    def _start_binance_streams(self, symbols: List[str]):
        """Start Binance WebSocket streams."""
        try:
            self.binance_manager = ThreadedWebSocketManager(
                api_key=self.config.BINANCE_API_KEY,
                api_secret=self.config.BINANCE_SECRET_KEY
            )
            self.binance_manager.start()
            
            # Subscribe to ticker streams
            for symbol in symbols:
                self.binance_manager.start_symbol_ticker_socket(
                    callback=self._handle_ticker_message,
                    symbol=symbol
                )
                
                # Subscribe to trade streams
                self.binance_manager.start_trade_socket(
                    callback=self._handle_trade_message,
                    symbol=symbol
                )
                
                # Subscribe to kline streams (1m, 5m, 1h)
                for interval in ['1m', '5m', '1h']:
                    self.binance_manager.start_kline_socket(
                        callback=self._handle_kline_message,
                        symbol=symbol,
                        interval=interval
                    )
                
                # Subscribe to depth streams (order book)
                self.binance_manager.start_depth_socket(
                    callback=self._handle_depth_message,
                    symbol=symbol
                )
            
            logging.info(f"Binance WebSocket streams started for {len(symbols)} symbols")
            
        except Exception as e:
            logging.error(f"Error starting Binance streams: {e}")
    
    def _start_public_streams(self, symbols: List[str]):
        """Start public WebSocket streams for additional data."""
        # CoinGecko doesn't have WebSocket, but we can simulate with periodic updates
        threading.Thread(
            target=self._periodic_coingecko_updates, 
            args=(symbols,), 
            daemon=True
        ).start()
    
    def _handle_ticker_message(self, msg):
        """Handle ticker price updates from Binance."""
        try:
            symbol = msg['s']
            price = float(msg['c'])
            volume = float(msg['v'])
            price_change_24h = float(msg['P'])
            
            # Store price data
            self.price_data[symbol].append({
                'timestamp': datetime.now(),
                'price': price,
                'volume': volume,
                'price_change_24h': price_change_24h
            })
            
            # Keep only last 1000 price points
            if len(self.price_data[symbol]) > 1000:
                self.price_data[symbol].popleft()
            
            # Trigger callbacks
            self._trigger_callbacks('price_update', {
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'price_change_24h': price_change_24h,
                'timestamp': datetime.now()
            })
            
            self.message_count += 1
            
        except Exception as e:
            logging.error(f"Error handling ticker message: {e}")
    
    def _handle_trade_message(self, msg):
        """Handle individual trade data from Binance."""
        try:
            symbol = msg['s']
            price = float(msg['p'])
            quantity = float(msg['q'])
            is_buyer_maker = msg['m']
            
            trade_data = {
                'timestamp': datetime.fromtimestamp(msg['T'] / 1000),
                'price': price,
                'quantity': quantity,
                'is_buyer_maker': is_buyer_maker
            }
            
            self.trade_data[symbol].append(trade_data)
            
            # Keep only last 100 trades
            if len(self.trade_data[symbol]) > 100:
                self.trade_data[symbol].popleft()
            
            # Calculate trade intensity
            recent_trades = list(self.trade_data[symbol])[-10:]  # Last 10 trades
            if len(recent_trades) >= 2:
                time_span = (recent_trades[-1]['timestamp'] - recent_trades[0]['timestamp']).total_seconds()
                if time_span > 0:
                    trade_intensity = len(recent_trades) / time_span  # trades per second
                    
                    self._trigger_callbacks('trade_intensity', {
                        'symbol': symbol,
                        'intensity': trade_intensity,
                        'recent_trades': len(recent_trades),
                        'timespan': time_span
                    })
            
            self.message_count += 1
            
        except Exception as e:
            logging.error(f"Error handling trade message: {e}")
    
    def _handle_kline_message(self, msg):
        """Handle kline (candlestick) data from Binance."""
        try:
            kline = msg['k']
            symbol = kline['s']
            interval = kline['i']
            is_closed = kline['x']  # True if this kline is closed
            
            candle_data = {
                'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'is_closed': is_closed
            }
            
            # Store candle data
            if is_closed:  # Only store completed candles
                self.candle_data[symbol][interval].append(candle_data)
                
                # Keep last 1000 candles
                if len(self.candle_data[symbol][interval]) > 1000:
                    self.candle_data[symbol][interval] = self.candle_data[symbol][interval][-1000:]
                
                # Trigger callbacks for completed candles
                self._trigger_callbacks('candle_update', {
                    'symbol': symbol,
                    'interval': interval,
                    'candle': candle_data
                })
            
            # Always trigger for current candle updates
            self._trigger_callbacks('candle_tick', {
                'symbol': symbol,
                'interval': interval,
                'candle': candle_data
            })
            
            self.message_count += 1
            
        except Exception as e:
            logging.error(f"Error handling kline message: {e}")
    
    def _handle_depth_message(self, msg):
        """Handle order book depth updates from Binance."""
        try:
            symbol = msg['s']
            
            # Update order book
            if symbol not in self.order_book_data:
                self.order_book_data[symbol] = {'bids': {}, 'asks': {}}
            
            # Update bids
            for bid in msg['b']:
                price = float(bid[0])
                quantity = float(bid[1])
                if quantity == 0:
                    self.order_book_data[symbol]['bids'].pop(price, None)
                else:
                    self.order_book_data[symbol]['bids'][price] = quantity
            
            # Update asks
            for ask in msg['a']:
                price = float(ask[0])
                quantity = float(ask[1])
                if quantity == 0:
                    self.order_book_data[symbol]['asks'].pop(price, None)
                else:
                    self.order_book_data[symbol]['asks'][price] = quantity
            
            # Calculate spread and liquidity metrics
            bids = self.order_book_data[symbol]['bids']
            asks = self.order_book_data[symbol]['asks']
            
            if bids and asks:
                best_bid = max(bids.keys())
                best_ask = min(asks.keys())
                spread = best_ask - best_bid
                spread_pct = (spread / best_ask) * 100
                
                # Calculate liquidity (sum of top 5 levels)
                top_bids = sorted(bids.items(), reverse=True)[:5]
                top_asks = sorted(asks.items())[:5]
                
                bid_liquidity = sum(qty for _, qty in top_bids)
                ask_liquidity = sum(qty for _, qty in top_asks)
                
                self._trigger_callbacks('orderbook_update', {
                    'symbol': symbol,
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'spread': spread,
                    'spread_pct': spread_pct,
                    'bid_liquidity': bid_liquidity,
                    'ask_liquidity': ask_liquidity,
                    'timestamp': datetime.now()
                })
            
            self.message_count += 1
            
        except Exception as e:
            logging.error(f"Error handling depth message: {e}")
    
    def _periodic_coingecko_updates(self, symbols: List[str]):
        """Periodically fetch CoinGecko data."""
        while self.is_running:
            try:
                # This would normally fetch from CoinGecko API
                # For now, just trigger a callback to indicate we should fetch
                self._trigger_callbacks('coingecko_update_needed', {
                    'symbols': symbols,
                    'timestamp': datetime.now()
                })
                
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logging.error(f"Error in periodic CoinGecko updates: {e}")
                time.sleep(30)
    
    def _data_processor(self):
        """Background data processing thread."""
        while self.is_running:
            try:
                # Calculate real-time indicators
                self._calculate_realtime_indicators()
                
                # Detect anomalies
                self._detect_price_anomalies()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                time.sleep(1)  # Process every second
                
            except Exception as e:
                logging.error(f"Error in data processor: {e}")
                time.sleep(5)
    
    def _calculate_realtime_indicators(self):
        """Calculate real-time technical indicators."""
        for symbol, prices in self.price_data.items():
            if len(prices) < 20:  # Need minimum data
                continue
            
            try:
                # Convert to lists for calculation
                price_list = [p['price'] for p in list(prices)[-50:]]  # Last 50 prices
                timestamps = [p['timestamp'] for p in list(prices)[-50:]]
                
                if len(price_list) >= 20:
                    # Simple Moving Average
                    sma_20 = np.mean(price_list[-20:])
                    
                    # Price momentum
                    if len(price_list) >= 2:
                        momentum = (price_list[-1] - price_list[-2]) / price_list[-2] * 100
                    else:
                        momentum = 0
                    
                    # Volatility (rolling std)
                    if len(price_list) >= 10:
                        volatility = np.std(price_list[-10:]) / np.mean(price_list[-10:]) * 100
                    else:
                        volatility = 0
                    
                    self._trigger_callbacks('realtime_indicators', {
                        'symbol': symbol,
                        'current_price': price_list[-1],
                        'sma_20': sma_20,
                        'momentum': momentum,
                        'volatility': volatility,
                        'timestamp': timestamps[-1]
                    })
            
            except Exception as e:
                logging.error(f"Error calculating indicators for {symbol}: {e}")
    
    def _detect_price_anomalies(self):
        """Detect unusual price movements."""
        for symbol, prices in self.price_data.items():
            if len(prices) < 10:
                continue
            
            try:
                recent_prices = [p['price'] for p in list(prices)[-10:]]
                
                if len(recent_prices) >= 5:
                    # Calculate recent average and std
                    avg_price = np.mean(recent_prices[:-1])  # Exclude latest price
                    std_price = np.std(recent_prices[:-1])
                    latest_price = recent_prices[-1]
                    
                    if std_price > 0:
                        z_score = (latest_price - avg_price) / std_price
                        
                        # Trigger anomaly alert for significant deviations
                        if abs(z_score) > 2.5:  # 2.5 standard deviations
                            self._trigger_callbacks('price_anomaly', {
                                'symbol': symbol,
                                'current_price': latest_price,
                                'average_price': avg_price,
                                'z_score': z_score,
                                'severity': 'HIGH' if abs(z_score) > 3 else 'MEDIUM',
                                'direction': 'UP' if z_score > 0 else 'DOWN',
                                'timestamp': datetime.now()
                            })
            
            except Exception as e:
                logging.error(f"Error detecting anomalies for {symbol}: {e}")
    
    def _update_performance_metrics(self):
        """Update streaming performance metrics."""
        current_time = datetime.now()
        time_diff = (current_time - self.last_message_time).total_seconds()
        
        if time_diff >= 60:  # Update every minute
            messages_per_second = self.message_count / max(time_diff, 1)
            
            self._trigger_callbacks('performance_update', {
                'messages_per_second': messages_per_second,
                'total_messages': self.message_count,
                'active_symbols': len(self.price_data),
                'uptime': time_diff,
                'timestamp': current_time
            })
            
            # Reset counters
            self.message_count = 0
            self.last_message_time = current_time
    
    def _trigger_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Trigger all callbacks for a specific event type."""
        for callback in self.callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                logging.error(f"Error in callback for {event_type}: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        if symbol in self.price_data and self.price_data[symbol]:
            return self.price_data[symbol][-1]['price']
        return None
    
    def get_price_history(self, symbol: str, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get price history for specified time period."""
        if symbol not in self.price_data:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            p for p in self.price_data[symbol] 
            if p['timestamp'] >= cutoff_time
        ]
    
    def get_candle_data(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Get candle data as DataFrame."""
        if symbol not in self.candle_data or interval not in self.candle_data[symbol]:
            return pd.DataFrame()
        
        candles = self.candle_data[symbol][interval][-limit:]
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def get_order_book_summary(self, symbol: str) -> Dict[str, Any]:
        """Get order book summary."""
        if symbol not in self.order_book_data:
            return {}
        
        bids = self.order_book_data[symbol]['bids']
        asks = self.order_book_data[symbol]['asks']
        
        if not bids or not asks:
            return {}
        
        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': best_ask - best_bid,
            'spread_pct': ((best_ask - best_bid) / best_ask) * 100,
            'bid_depth': sum(bids.values()),
            'ask_depth': sum(asks.values()),
            'total_levels': len(bids) + len(asks)
        }
    
    def get_trade_summary(self, symbol: str, minutes: int = 5) -> Dict[str, Any]:
        """Get recent trade summary."""
        if symbol not in self.trade_data:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_trades = [
            t for t in self.trade_data[symbol]
            if t['timestamp'] >= cutoff_time
        ]
        
        if not recent_trades:
            return {}
        
        total_volume = sum(t['quantity'] for t in recent_trades)
        buy_volume = sum(t['quantity'] for t in recent_trades if not t['is_buyer_maker'])
        sell_volume = total_volume - buy_volume
        
        return {
            'trade_count': len(recent_trades),
            'total_volume': total_volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_ratio': buy_volume / total_volume if total_volume > 0 else 0,
            'avg_trade_size': total_volume / len(recent_trades) if recent_trades else 0,
            'time_period_minutes': minutes
        }
    
    def is_streaming_active(self) -> bool:
        """Check if streaming is currently active."""
        return self.is_running
    
    def get_stream_status(self) -> Dict[str, Any]:
        """Get comprehensive streaming status."""
        return {
            'is_running': self.is_running,
            'active_symbols': list(self.price_data.keys()),
            'message_count': self.message_count,
            'last_update': max([
                max(prices)['timestamp'] for prices in self.price_data.values()
                if prices
            ]) if self.price_data else None,
            'binance_connected': self.binance_manager is not None and self.binance_manager._running,
            'websocket_connections': len(self.websocket_connections),
            'data_points': {
                'prices': sum(len(prices) for prices in self.price_data.values()),
                'trades': sum(len(trades) for trades in self.trade_data.values()),
                'candles': sum(
                    sum(len(intervals) for intervals in symbol_data.values())
                    for symbol_data in self.candle_data.values()
                ),
                'orderbooks': len(self.order_book_data)
            }
        }