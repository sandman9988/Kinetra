#!/usr/bin/env python3
"""
MT5 Bridge Server for WSL2

Run this script on Windows with MT5 terminal open.
It serves symbol specs and tick data to WSL2 clients.

Usage:
    python mt5_bridge_server.py [--port 5555]
"""

import socket
import json
import threading
from datetime import datetime
import MetaTrader5 as mt5


class MT5BridgeServer:
    def __init__(self, port: int = 5555):
        self.port = port
        self.running = False
        self.mt5_connected = False

    def connect_mt5(self) -> bool:
        if not mt5.initialize():
            print(f"MT5 init failed: {mt5.last_error()}")
            return False
        self.mt5_connected = True
        info = mt5.account_info()
        print(f"Connected: {info.server} - {info.login}")
        return True

    def get_symbol_spec(self, symbol: str) -> dict:
        info = mt5.symbol_info(symbol)
        if not info:
            return {"error": f"Symbol {symbol} not found"}

        tick = mt5.symbol_info_tick(symbol)
        spread = (tick.ask - tick.bid) / info.point if tick else 0

        return {
            "symbol": symbol,
            "digits": info.digits,
            "point": info.point,
            "contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "spread_current": spread,
            "swap_long": info.swap_long,
            "swap_short": info.swap_short,
            "bid": tick.bid if tick else 0,
            "ask": tick.ask if tick else 0,
        }

    def get_rates(self, symbol: str, timeframe: str, count: int) -> list[dict[str, int | float]] | dict[str, str]:
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
        }
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None:
            return {"error": f"No rates for {symbol}"}
        return [{
            "time": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "tick_volume": int(r[5]),
            "spread": int(r[6]),
            "real_volume": int(r[7]),
        } for r in rates]

    def handle_client(self, conn, addr):
        print(f"Client connected: {addr}")
        try:
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break

                try:
                    request = json.loads(data.decode())
                    cmd = request.get("cmd", "")

                    if cmd == "symbol_info":
                        response = self.get_symbol_spec(request.get("symbol", ""))
                    elif cmd == "tick":
                        tick = mt5.symbol_info_tick(request.get("symbol", ""))
                        response = {"bid": tick.bid, "ask": tick.ask} if tick else {"error": "No tick"}
                    elif cmd == "account":
                        info = mt5.account_info()
                        response = {"balance": info.balance, "equity": info.equity} if info else {"error": "No account"}
                    elif cmd == "rates":
                        response = self.get_rates(
                            request.get("symbol", "EURUSD"),
                            request.get("timeframe", "H1"),
                            request.get("count", 100)
                        )
                    elif cmd == "symbols":
                        symbols = mt5.symbols_get()
                        response = [s.name for s in symbols] if symbols else []
                    elif cmd == "ping":
                        response = {"status": "ok", "time": datetime.now().isoformat()}
                    else:
                        response = {"error": f"Unknown command: {cmd}"}

                    conn.sendall(json.dumps(response).encode())
                except json.JSONDecodeError:
                    conn.sendall(json.dumps({"error": "Invalid JSON"}).encode())
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            conn.close()
            print(f"Client disconnected: {addr}")

    def run(self):
        if not self.connect_mt5():
            return

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", self.port))
        server.listen(5)

        self.running = True
        print(f"MT5 Bridge Server running on port {self.port}")
        print("Press Ctrl+C to stop")

        try:
            while self.running:
                conn, addr = server.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.running = False
            server.close()
            mt5.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    server = MT5BridgeServer(port=args.port)
    server.run()
