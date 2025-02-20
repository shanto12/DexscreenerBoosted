import requests
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import time
from tabulate import tabulate
from contextlib import contextmanager
import functools

# Constants
BASE_URL = "https://api.dexscreener.com"
BOOSTS_ENDPOINT = f"{BASE_URL}/token-boosts/latest/v1"
TOKENS_PAIRS_ENDPOINT = f"{BASE_URL}/token-pairs/v1"
PAIRS_ENDPOINT = f"{BASE_URL}/latest/dex/pairs"
HEADERS = {"User-Agent": "BoostedTokenAnalyzer/1.0", "Accept": "application/json"}
INITIAL_CAPITAL = 10000.0  # Changed to float
BUY_PERCENTAGE = 0.10  # Changed to float
SELL_THRESHOLD = 1.4  # Changed to float
STOP_LOSS_THRESHOLD = 0.7  # Changed to float
CHAIN_ID = "solana"
FIRST_RUN = [True]  # Mutable singleton to modify in main()


@functools.lru_cache(maxsize=128)
def cached_api_get(endpoint: str) -> Optional[Dict]:
    """Cached API request handler"""
    try:
        response = requests.get(endpoint, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error making request to {endpoint}: {str(e)}")
        return None


def get_boosted_tokens() -> Optional[List[Dict]]:
    """Fetch latest boosted tokens"""
    return cached_api_get(BOOSTS_ENDPOINT)


def get_token_pairs(chain_id: str, token_address: str) -> Optional[List[Dict]]:
    """Fetch pair data for a token"""
    return cached_api_get(f"{TOKENS_PAIRS_ENDPOINT}/{chain_id}/{token_address}")


def get_pair_data(chain_id: str, pair_address: str) -> Optional[Dict]:
    """Fetch detailed pair data"""
    return cached_api_get(f"{PAIRS_ENDPOINT}/{chain_id}/{pair_address}")


def format_price(price: any) -> str:
    """Format price with up to 10 decimal places"""
    if not isinstance(price, (int, float)) or price == "N/A":
        return "N/A"
    return f"{float(price):.10f}".rstrip('0').rstrip('.')


def process_token_data(token_data: List[Dict]) -> List[Dict]:
    """Process token data with deduplication"""
    current_time = int(time.time())
    seen_tokens: Dict[Tuple[str, str], Dict] = {}

    for token in token_data:
        if token.get("chainId") != CHAIN_ID:
            continue

        pair_address = token.get("pairAddress") or "N/A"
        boost_time = token.get("boostTimestamp", current_time)

        if pair_address == "N/A":
            token_pairs = get_token_pairs(CHAIN_ID, token.get("tokenAddress", ""))
            pair_address = token_pairs[0].get("pairAddress", "N/A") if token_pairs else "N/A"

        pair_data = get_pair_data(CHAIN_ID, pair_address) if pair_address != "N/A" else None
        if pair_data and pair_data.get("pairs"):
            boost_time = pair_data["pairs"][0].get("pairCreatedAt", current_time * 1000) / 1000

        age_minutes = max(0, current_time - boost_time) / 60
        token_info = {
            "age_minutes": age_minutes,
            "token_address": token.get("tokenAddress", "N/A"),
            "chain_id": token.get("chainId", "N/A"),
            "boost_amount": token.get("amount", 0),
            "total_boost_amount": token.get("totalAmount", 0),
            "pair_address": pair_address,
            "token_name": token.get("description", "Unknown").split()[0] if token.get("description") else "Unknown",
            "price_usd": "N/A",
            "liquidity_usd": 0,
            "fdv": 0,
            "market_cap": 0,
            "volume_24h": 0,
            "txns_24h": 0,
            "buys_24h": 0,
            "sells_24h": 0,
            "buy_volume_24h": 0,
            "sell_volume_24h": 0,
            "price_change_5m": 0,
            "price_change_1h": 0,
            "price_change_6h": 0,
            "price_change_24h": 0
        }

        if pair_data and pair_data.get("pairs"):
            pair = pair_data["pairs"][0]
            token_info.update({
                "pair_address": pair.get("pairAddress", "N/A"),
                "token_name": pair.get("baseToken", {}).get("name", token_info["token_name"]),
                "price_usd": float(pair.get("priceUsd", "N/A")) if pair.get("priceUsd", "N/A") != "N/A" else "N/A",
                "liquidity_usd": pair.get("liquidity", {}).get("usd", 0),
                "fdv": pair.get("fdv", 0),
                "market_cap": pair.get("marketCap", 0),
                "volume_24h": pair.get("volume", {}).get("h24", 0)
            })

            txns = pair.get("txns", {}).get("h24", {})
            token_info.update({
                "txns_24h": txns.get("buys", 0) + txns.get("sells", 0),
                "buys_24h": txns.get("buys", 0),
                "sells_24h": txns.get("sells", 0)
            })

            volume = pair.get("volume", {})
            total_volume = volume.get("h24", 0)
            token_info.update({
                "buy_volume_24h": volume.get("buyH24", 0) or (
                    token_info["buys_24h"] / token_info["txns_24h"] * total_volume if token_info[
                                                                                          "txns_24h"] > 0 else 0),
                "sell_volume_24h": volume.get("sellH24", 0) or (
                    token_info["sells_24h"] / token_info["txns_24h"] * total_volume if token_info[
                                                                                           "txns_24h"] > 0 else 0),
                "price_change_5m": pair.get("priceChange", {}).get("m5", 0),
                "price_change_1h": pair.get("priceChange", {}).get("h1", 0),
                "price_change_6h": pair.get("priceChange", {}).get("h6", 0),
                "price_change_24h": pair.get("priceChange", {}).get("h24", 0)
            })

        key = (token_info["token_address"], token_info["pair_address"])
        if key not in seen_tokens or token_info["boost_amount"] > seen_tokens[key]["boost_amount"]:
            seen_tokens[key] = token_info

    return sorted(seen_tokens.values(), key=lambda x: x["age_minutes"])


def display_tokens(tokens: List[Dict], current_capital: float, trades: Dict, iteration: int,
                   runtime_minutes: float) -> None:
    """Display token information and trading status"""
    table_data = [[
        f"{t['age_minutes']:.2f}m",
        t.get("token_name", "Unknown")[:20],
        t['token_address'][:45],
        t.get("pair_address", "N/A")[:45],
        f"{t['boost_amount']:,}",
        format_price(t["price_usd"]),
        f"${t['liquidity_usd']:,.0f}",
        f"${t['fdv']:,.0f}",
        f"${t['market_cap']:,.0f}",
        t['txns_24h'],
        t['buys_24h'],
        t['sells_24h'],
        f"${t['buy_volume_24h']:,.0f}",
        f"${t['sell_volume_24h']:,.0f}",
        f"{t['price_change_5m']:.2f}%",
        f"{t['price_change_1h']:.2f}%",
        f"{t['price_change_6h']:.2f}%",
        f"{t['price_change_24h']:.2f}%",
        t['chain_id']
    ] for t in tokens[:10]]

    headers = ["Age (m)", "Name", "Token Address", "Pair Address", "Boost", "Price USD", "Liquidity USD", "FDV",
               "Market Cap",
               "Txns (24h)", "Buys (24h)", "Sells (24h)", "Buy Vol (24h)", "Sell Vol (24h)", "Price Chg 5m",
               "Price Chg 1h",
               "Price Chg 6h", "Price Chg 24h", "Chain"]

    print(f"\n=== Boosted Tokens - Iteration {iteration}, Runtime: {runtime_minutes:.2f} minutes ===")
    print(tabulate(sorted(table_data, key=lambda x: float(x[0][:-1])), headers=headers, tablefmt="pretty",
                   maxcolwidths=[10, 20, 45, 45, 10, 12, 15, 15, 15, 10, 10, 10, 15, 15, 10, 10, 10, 10, 10]))
    print(f"\nCurrent Capital: ${current_capital:,.2f}")

    current_time = int(time.time())
    for trade_type, headers, data_func in [
        ("buys", ["Token Name", "Token Address", "Buy Price USD", "Current Price USD", "Quantity", "Capital Spent",
                  "Current Value", "% Change", "PnL", "Highest Price USD", "Hold Age (m)", "% Diff to High"],
         lambda t: [t['token_name'][:20], t['token_address'][:45], format_price(t['buy_price']),
                    format_price(t['current_price']), f"{t['quantity']:.2f}",
                    f"${t['buy_price'] * t['quantity']:,.2f}", f"${t['current_price'] * t['quantity']:,.2f}",
                    f"{((t['current_price'] - t['buy_price']) / t['buy_price'] * 100) if t['buy_price'] > 0 else 0:.2f}%",
                    f"${(t['current_price'] - t['buy_price']) * t['quantity']:,.2f}", format_price(t['highest_price']),
                    f"{(current_time - t.get('buy_time', current_time)) / 60:.2f}m",
                    f"{((t['highest_price'] - t['current_price']) / t['highest_price'] * 100) if t['highest_price'] > 0 else 0:.2f}%"]),
        ("sells", ["Token Name", "Token Address", "Buy Price USD", "Sell Price USD", "Capital Spent", "Sell Value",
                   "PnL", "Highest Price USD", "Hold Age (m)"],
         lambda t: [t['token_name'][:20], t['token_address'][:45], format_price(t['buy_price']),
                    format_price(t['sell_price']), f"${t['buy_price'] * t['quantity']:,.2f}",
                    f"${t['sell_price'] * t['quantity']:,.2f}", f"${t['profit_loss']:.2f}",
                    format_price(t['highest_price']),
                    f"{(t.get('sell_time', current_time) - t.get('buy_time', current_time)) / 60:.2f}m"]),
        ("tracked_sold", ["Token Name", "Token Address", "Buy Price USD", "Sell Price USD", "Current Price USD",
                          "Highest Price USD", "% Apprec to High"],
         lambda t: [t['token_name'][:20], t['token_address'][:45], format_price(t['buy_price']),
                    format_price(t['sell_price']), format_price(t.get('current_price', t['highest_price'])),
                    format_price(t['highest_price']),
                    f"{((t['highest_price'] - t['buy_price']) / t['buy_price'] * 100) if t['buy_price'] > 0 else 0:.2f}%"])
    ]:
        if trades.get(trade_type):
            print(f"\n=== {trade_type.capitalize()} Trades ===")
            print(tabulate([data_func(t) for t in trades[trade_type]], headers=headers, tablefmt="pretty",
                           maxcolwidths=[20, 45] + [15] * (len(headers) - 2)))


def calculate_profit_loss(trades: Dict) -> Tuple[float, float, float]:
    """Calculate total profit/loss and unrealized PnL"""
    total_profit_loss = sum(trade["profit_loss"] for trade in trades["sells"])
    current_value = sum(trade["current_price"] * trade["quantity"] for trade in trades["buys"])
    capital_spent = sum(trade["buy_price"] * trade["quantity"] for trade in trades["buys"])
    return total_profit_loss, current_value, current_value - capital_spent if trades["buys"] else 0.0


def calculate_avg_percent_increase(trades: Dict, processed_tokens: List[Dict]) -> Tuple[float, List[str]]:
    """Calculate average percentage increase to highest price"""
    percent_increases, tokens_to_remove = [], []

    for trade_list in ["buys", "sells", "tracked_sold"]:
        for trade in trades.get(trade_list, [])[:]:
            token = next((t for t in processed_tokens if t["token_address"] == trade["token_address"]), None)
            current_price = token["price_usd"] if token and token["price_usd"] != "N/A" else trade["highest_price"]

            if current_price <= trade["buy_price"] * 0.8:
                tokens_to_remove.append(trade["token_address"])
                if trade_list != "sells":
                    trades[trade_list].remove(trade)
            else:
                percent_increases.append(((trade["highest_price"] - trade["buy_price"]) / trade["buy_price"]) * 100)

    return sum(percent_increases) / len(percent_increases) if percent_increases else 0.0, tokens_to_remove


@contextmanager
def trading_data_file():
    """Context manager for trading data file operations"""
    try:
        with open("trading_data.json", "r") as f:
            data = json.load(f)
            for trade in data.get("buys", []) + data.get("sells", []) + data.get("tracked_sold", []):
                trade.setdefault("highest_price", trade.get("sell_price", trade["buy_price"]))
                trade.setdefault("buy_time", int(time.time()))
                if "sells" in trade:
                    trade.setdefault("sell_time", int(time.time()))
                trade.setdefault("quantity", 1.0)
        yield data
    except FileNotFoundError:
        yield {"capital": INITIAL_CAPITAL, "buys": [], "sells": [], "tracked_sold": [], "known_tokens": []}
    finally:
        with open("trading_data.json", "w") as f:
            json.dump(data, f, indent=2)


def main():
    with trading_data_file() as trading_data:
        current_capital = float(trading_data["capital"])
        known_tokens = set(trading_data["known_tokens"])
        start_time, iteration = time.time(), 0

        while current_capital > 0:
            iteration += 1
            runtime_minutes = (time.time() - start_time) / 60
            print(
                f"\n=== Running at {datetime.now():%Y-%m-%d %H:%M:%S} - Iteration {iteration}, Runtime: {runtime_minutes:.2f} minutes ===")

            boosted_tokens = get_boosted_tokens()
            if not boosted_tokens:
                print("Failed to fetch boosted tokens. Retrying in 60 seconds...")
                time.sleep(60)
                continue

            processed_tokens = process_token_data(boosted_tokens)

            if FIRST_RUN[0]:
                print("Skipping buys on first run...")
                FIRST_RUN[0] = False
            else:
                for token in [t for t in processed_tokens if
                              t["token_address"] not in known_tokens and t["price_usd"] != "N/A"]:
                    buy_amount = current_capital * BUY_PERCENTAGE
                    price_usd = float(token["price_usd"])
                    quantity = buy_amount / price_usd
                    if quantity > 0:
                        print(
                            f"Buying {token['token_name']} at ${format_price(price_usd)} with ${buy_amount:,.2f} ({quantity:.2f} units, CA: {token['token_address']})")
                        current_capital -= buy_amount
                        trading_data["buys"].append({
                            "token_name": token["token_name"],
                            "token_address": token["token_address"],
                            "buy_price": price_usd,
                            "quantity": quantity,
                            "current_price": price_usd,
                            "highest_price": price_usd,
                            "buy_time": int(time.time())
                        })

            for trade in trading_data["buys"][:]:
                token = next((t for t in processed_tokens if t["token_address"] == trade["token_address"]), None)
                if token and token["price_usd"] != "N/A":
                    current_price = float(token["price_usd"])
                    trade["current_price"] = current_price
                    trade["highest_price"] = max(trade["highest_price"], current_price)

                    for threshold, action in [(SELL_THRESHOLD, "Selling"), (STOP_LOSS_THRESHOLD, "Stop Loss: Selling")]:
                        target = trade["buy_price"] * threshold
                        if (threshold > 1 and current_price >= target) or (threshold < 1 and current_price <= target):
                            sell_value = current_price * trade["quantity"]
                            profit_loss = sell_value - (trade["buy_price"] * trade["quantity"])
                            print(
                                f"{action} {trade['token_name']} at ${format_price(current_price)} (Bought at ${format_price(trade['buy_price'])}, Profit/Loss: ${profit_loss:.2f}, CA: {trade['token_address']})")
                            current_capital += sell_value
                            trade.update({"sell_price": current_price, "profit_loss": profit_loss,
                                          "sell_time": int(time.time())})
                            trading_data["sells"].append(trade)
                            trading_data["tracked_sold"].append(
                                {k: v for k, v in trade.items() if k != "current_price"})
                            trading_data["buys"].remove(trade)
                            break

            for tracked in trading_data.get("tracked_sold", [])[:]:
                token = next((t for t in processed_tokens if t["token_address"] == tracked["token_address"]), None)
                if token and token["price_usd"] != "N/A":
                    current_price = float(token["price_usd"])
                    tracked["highest_price"] = max(tracked["highest_price"], current_price)
                    if current_price <= tracked["buy_price"] * 0.8:
                        print(
                            f"Stopping tracking {tracked['token_name']} (CA: {tracked['token_address']}) - Price dropped 80% from buy price (${tracked['buy_price']} to ${current_price})")
                        trading_data["tracked_sold"].remove(tracked)

            avg_percent_increase, tokens_to_remove = calculate_avg_percent_increase(trading_data, processed_tokens)
            known_tokens.difference_update(tokens_to_remove)
            known_tokens.update(t["token_address"] for t in processed_tokens)
            trading_data["known_tokens"] = list(known_tokens)
            trading_data["capital"] = current_capital

            display_tokens(processed_tokens, current_capital, trading_data, iteration, runtime_minutes)

            total_profit_loss, current_value, unrealized_pnl = calculate_profit_loss(trading_data)
            print(f"\n=== Trading Summary - Iteration {iteration} ===")
            print(f"Total Capital: ${current_capital:,.2f}\nCurrent Holdings Value: ${current_value:,.2f}")
            print(
                f"Total Realized Profit/Loss: ${total_profit_loss:,.2f}\nUnrealized Profit/Loss: ${unrealized_pnl:,.2f}")
            print(
                f"Number of Active Trades: {len(trading_data['buys'])}\nNumber of Completed Trades: {len(trading_data['sells'])}")
            print(f"Number of Tracked Sold Tokens: {len(trading_data.get('tracked_sold', []))}")
            print(f"Average % Increase to Highest Price: {avg_percent_increase:.2f}%")

            time.sleep(60)

        runtime_minutes = (time.time() - start_time) / 60
        print(
            f"\n=== All capital lost. Simulation ended. ===\nFinal Report - Iteration {iteration}, Runtime: {runtime_minutes:.2f} minutes:")
        for k, v in {"Final Capital": current_capital, "Current Holdings Value": current_value,
                     "Total Realized Profit/Loss": total_profit_loss, "Unrealized Profit/Loss": unrealized_pnl}.items():
            print(f"{k}: ${v:,.2f}")
        print(
            f"Number of Active Trades: {len(trading_data['buys'])}\nNumber of Completed Trades: {len(trading_data['sells'])}")
        print(f"Number of Tracked Sold Tokens: {len(trading_data.get('tracked_sold', []))}")
        print(
            f"Average % Increase to Highest Price: {calculate_avg_percent_increase(trading_data, processed_tokens)[0]:.2f}%")


if __name__ == "__main__":
    main()