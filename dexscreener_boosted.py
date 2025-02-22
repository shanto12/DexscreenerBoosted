import requests
import json
from datetime import datetime, timedelta
import time
from tabulate import tabulate

# Constants
BASE_URL = "https://api.dexscreener.com"
BOOSTS_ENDPOINT = f"{BASE_URL}/token-boosts/latest/v1"
TOKENS_PAIRS_ENDPOINT = f"{BASE_URL}/token-pairs/v1"
PAIRS_ENDPOINT = f"{BASE_URL}/latest/dex/pairs"
HEADERS = {
    "User-Agent": "BoostedTokenAnalyzer/1.0",
    "Accept": "application/json"
}

# Trading parameters
INITIAL_CAPITAL = 10000  # $10,000 starting capital
BUY_PERCENTAGE = 0.10  # Buy with 10% of available capital
SELL_THRESHOLD = 1.4  # Sell when price reaches 40% appreciation
STOP_LOSS_THRESHOLD = 0.7  # Sell when price falls to 30% below purchase price
CHAIN_ID = "solana"  # Focus on Solana tokens
FIRST_RUN = True  # Flag to skip buying on the first run

def get_boosted_tokens():
    """Fetch the latest boosted tokens from DEX Screener API"""
    try:
        response = requests.get(BOOSTS_ENDPOINT, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching data: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error making request: {str(e)}")
        return None

def get_token_pairs(chain_id, token_address):
    """Fetch pair data for a specific token address"""
    try:
        endpoint = f"{TOKENS_PAIRS_ENDPOINT}/{chain_id}/{token_address}"
        response = requests.get(endpoint, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching token pairs: {str(e)}")
        return None

def get_pair_data(chain_id, pair_address):
    """Fetch detailed pair data for a specific pair address"""
    try:
        endpoint = f"{PAIRS_ENDPOINT}/{chain_id}/{pair_address}"
        response = requests.get(endpoint, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching pair data: {str(e)}")
        return None

def format_price(price):
    """Format price to show up to 10 decimal places, removing trailing zeros and decimal if no significant digits"""
    if price == "N/A" or not isinstance(price, (int, float)):
        return "N/A"
    return f"{price:.10f}".rstrip('0').rstrip('.')

def process_token_data(token_data):
    """Process token data and extract relevant information, deduplicating based on token_address and pair_address"""
    processed_tokens = []
    current_time = int(time.time())
    seen_tokens = {}

    for token in token_data:
        if token.get("chainId") != CHAIN_ID:
            continue

        boost_time = token.get("boostTimestamp", None)
        pair_address = token.get("pairAddress", None)

        if not pair_address:
            token_pairs = get_token_pairs(token.get("chainId", "solana"), token.get("tokenAddress", ""))
            if token_pairs and len(token_pairs) > 0:
                pair_address = token_pairs[0].get("pairAddress", "N/A")

        pair_data = get_pair_data(token.get("chainId", "solana"), pair_address) if pair_address and pair_address != "N/A" else None

        if pair_data and pair_data.get("pairs") and len(pair_data["pairs"]) > 0:
            primary_pair = pair_data["pairs"][0]
            boost_time = primary_pair.get("pairCreatedAt", current_time) / 1000

        age_seconds = max(0, current_time - (boost_time if isinstance(boost_time, (int, float)) else current_time))
        age_minutes = age_seconds / 60

        token_info = {
            "age_minutes": age_minutes,
            "token_address": token.get("tokenAddress", "N/A"),
            "chain_id": token.get("chainId", "N/A"),
            "boost_amount": token.get("amount", 0),
            "total_boost_amount": token.get("totalAmount", 0),
            "pair_address": pair_address if pair_address and pair_address != "N/A" else "N/A",
            "token_name": (token.get("description", "").strip().split() or ["Unknown"])[0],
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

        if pair_data and pair_data.get("pairs") and len(pair_data["pairs"]) > 0:
            primary_pair = pair_data["pairs"][0]
            token_info["pair_address"] = primary_pair.get("pairAddress", "N/A")
            token_info["token_name"] = primary_pair.get("baseToken", {}).get("name", token_info["token_name"])

            price_usd = primary_pair.get("priceUsd", "N/A")
            if price_usd != "N/A" and isinstance(price_usd, str):
                try:
                    token_info["price_usd"] = float(price_usd)
                except ValueError:
                    token_info["price_usd"] = "N/A"
                    print(f"Warning: Invalid priceUsd format for token {token_info['token_name']} - {token_info['token_address']}")
            elif isinstance(price_usd, (int, float)):
                token_info["price_usd"] = price_usd
            else:
                token_info["price_usd"] = "N/A"
                print(f"Warning: priceUsd not found or invalid for token {token_info['token_name']} - {token_info['token_address']}")

            token_info["liquidity_usd"] = primary_pair.get("liquidity", {}).get("usd", 0)
            token_info["fdv"] = primary_pair.get("fdv", 0)
            token_info["market_cap"] = primary_pair.get("marketCap", 0)
            token_info["volume_24h"] = primary_pair.get("volume", {}).get("h24", 0)

            txns_data = primary_pair.get("txns", {})
            token_info["txns_24h"] = txns_data.get("h24", {}).get("buys", 0) + txns_data.get("h24", {}).get("sells", 0)
            token_info["buys_24h"] = txns_data.get("h24", {}).get("buys", 0)
            token_info["sells_24h"] = txns_data.get("h24", {}).get("sells", 0)

            volume_data = primary_pair.get("volume", {})
            token_info["buy_volume_24h"] = volume_data.get("buyH24", 0)
            token_info["sell_volume_24h"] = volume_data.get("sellH24", 0)
            total_volume_h24 = volume_data.get("h24", 0)
            total_txns_h24 = token_info["buys_24h"] + token_info["sells_24h"]
            if total_txns_h24 > 0 and total_volume_h24 > 0 and (token_info["buy_volume_24h"] == 0 or token_info["sell_volume_24h"] == 0):
                token_info["buy_volume_24h"] = (token_info["buys_24h"] / total_txns_h24) * total_volume_h24
                token_info["sell_volume_24h"] = (token_info["sells_24h"] / total_txns_h24) * total_volume_h24

            price_change_data = primary_pair.get("priceChange", {})
            token_info["price_change_5m"] = price_change_data.get("m5", 0)
            token_info["price_change_1h"] = price_change_data.get("h1", 0)
            token_info["price_change_6h"] = price_change_data.get("h6", 0)
            token_info["price_change_24h"] = price_change_data.get("h24", 0)

        if token_info["age_minutes"] == 0:
            print(f"Warning: Age is 0 for token {token_info['token_name']} - {token_info['token_address']}")
            print(f"Boost Time: {boost_time}, Current Time: {current_time}")

        key = (token_info["token_address"], token_info["pair_address"])
        if key not in seen_tokens:
            seen_tokens[key] = token_info
        else:
            existing_token = seen_tokens[key]
            if token_info["boost_amount"] > existing_token["boost_amount"] or \
               (token_info["boost_amount"] == existing_token["boost_amount"] and token_info["age_minutes"] < existing_token["age_minutes"]):
                seen_tokens[key] = token_info

    processed_tokens = list(seen_tokens.values())
    processed_tokens.sort(key=lambda x: x["age_minutes"])
    return processed_tokens

def display_tokens(tokens, current_capital, trades, iteration, runtime_minutes):
    """Display token information, trading status, and active/completed trades in tables"""
    table_data = []
    for token in tokens[:10]:
        price_usd_str = format_price(token["price_usd"])
        row = [
            f"{token['age_minutes']:.2f}m",
            token.get("token_name", "Unknown")[:20],
            token['token_address'][:45],
            token.get("pair_address", "N/A")[:45],
            f"{token['boost_amount']:,}",
            price_usd_str,
            f"${token['liquidity_usd']:,.0f}",
            f"${token['fdv']:,.0f}",
            f"${token['market_cap']:,.0f}",
            token['txns_24h'],
            token['buys_24h'],
            token['sells_24h'],
            f"${token['buy_volume_24h']:,.0f}",
            f"${token['sell_volume_24h']:,.0f}",
            f"{token['price_change_5m']:.2f}%",
            f"{token['price_change_1h']:.2f}%",
            f"{token['price_change_6h']:.2f}%",
            f"{token['price_change_24h']:.2f}%",
            token['chain_id']
        ]
        table_data.append(row)

    table_data.sort(key=lambda x: float(x[0].replace("m", "")))

    headers = ["Age (m)", "Name", "Token Address", "Pair Address", "Boost", "Price USD", "Liquidity USD", "FDV",
               "Market Cap", "Txns (24h)", "Buys (24h)", "Sells (24h)", "Buy Vol (24h)", "Sell Vol (24h)",
               "Price Chg 5m", "Price Chg 1h", "Price Chg 6h", "Price Chg 24h", "Chain"]
    print(f"\n=== Boosted Tokens (Sorted by Age - Newest First) - Iteration {iteration}, Runtime: {runtime_minutes:.2f} minutes ===")
    print(tabulate(table_data, headers=headers, tablefmt="pretty",
                   maxcolwidths=[10, 20, 45, 45, 10, 12, 15, 15, 15, 10, 10, 10, 15, 15, 10, 10, 10, 10, 10]))

    print(f"\nCurrent Capital: ${current_capital:,.2f}")

    if trades['buys']:
        active_trades_data = []
        current_time = int(time.time())
        for trade in trades['buys']:
            capital_spent = trade['buy_price'] * trade['quantity']
            current_value = trade['current_price'] * trade['quantity']
            percent_change = ((trade['current_price'] - trade['buy_price']) / trade['buy_price']) * 100 if trade['buy_price'] > 0 else 0.0
            current_pnl = current_value - capital_spent
            buy_time = trade.get('buy_time', current_time)
            age_minutes = (current_time - buy_time) / 60
            percent_diff_to_highest = ((trade['highest_price'] - trade['current_price']) / trade['highest_price']) * 100 if trade['highest_price'] > 0 else 0.0
            # Estimate last buy minutes using price_change_5m
            token = next((t for t in tokens if t["token_address"] == trade["token_address"]), None)
            if token:
                if token["price_change_5m"] == 0:
                    last_buy_minutes = 5.0  # No change in 5 minutes, assume 5+ minutes ago
                else:
                    # Estimate based on magnitude of price change (smaller change = longer ago, within 5 minutes)
                    change_magnitude = abs(token["price_change_5m"])
                    last_buy_minutes = min(5.0, 5.0 / (1 + change_magnitude))  # Scales from ~0 to 5 minutes
            else:
                last_buy_minutes = 5.0  # Default if no token data
            active_trades_data.append([
                trade['token_name'][:20],
                trade['token_address'][:45],
                format_price(trade['buy_price']),
                format_price(trade['current_price']),
                f"{trade['quantity']:.2f}",
                f"${capital_spent:,.2f}",
                f"${current_value:,.2f}",
                f"{percent_change:.2f}%",
                f"${current_pnl:,.2f}",
                format_price(trade['highest_price']),
                f"{age_minutes:.2f}m",
                f"{percent_diff_to_highest:.2f}%",
                f"{last_buy_minutes:.2f}m"
            ])

        active_trades_headers = ["Token Name", "Token Address", "Buy Price USD", "Current Price USD", "Quantity",
                                 "Capital Spent", "Current Value", "% Change", "PnL", "Highest Price USD",
                                 "Hold Age (m)", "% Diff to High", "Last Buy (m)"]
        print("\n=== Active Trades ===")
        print(tabulate(active_trades_data, headers=active_trades_headers, tablefmt="pretty",
                       maxcolwidths=[20, 45, 15, 15, 10, 15, 15, 10, 15, 15, 10, 15, 10]))

    if trades['sells']:
        completed_trades_data = []
        for trade in trades['sells']:
            capital_spent = trade['buy_price'] * trade['quantity'] if 'quantity' in trade else trade['buy_price']
            buy_time = trade.get('buy_time', int(time.time()))
            sell_time = trade.get('sell_time', int(time.time()))
            hold_age_minutes = (sell_time - buy_time) / 60
            completed_trades_data.append([
                trade['token_name'][:20],
                trade['token_address'][:45],
                format_price(trade['buy_price']),
                format_price(trade['sell_price']),
                f"${capital_spent:,.2f}",
                f"${(trade['sell_price'] * trade['quantity'] if 'quantity' in trade else trade['sell_price']):,.2f}",
                f"${trade['profit_loss']:.2f}",
                format_price(trade['highest_price']),
                f"{hold_age_minutes:.2f}m"
            ])

        completed_trades_headers = ["Token Name", "Token Address", "Buy Price USD", "Sell Price USD", "Capital Spent",
                                    "Sell Value", "PnL", "Highest Price USD", "Hold Age (m)"]
        print("\n=== Completed Trades ===")
        print(tabulate(completed_trades_data, headers=completed_trades_headers, tablefmt="pretty",
                       maxcolwidths=[20, 45, 15, 15, 15, 15, 15, 15, 15]))

    if trades.get('tracked_sold', []):
        tracked_sold_data = []
        for tracked_trade in trades['tracked_sold']:
            token_address = tracked_trade['token_address']
            token = next((t for t in tokens if t["token_address"] == token_address), None)
            current_price = token["price_usd"] if token and token["price_usd"] != "N/A" and isinstance(token["price_usd"], (int, float)) else tracked_trade["highest_price"]
            tracked_trade["highest_price"] = max(tracked_trade["highest_price"], current_price)
            tracked_sold_data.append([
                tracked_trade['token_name'][:20],
                tracked_trade['token_address'][:45],
                format_price(tracked_trade['buy_price']),
                format_price(tracked_trade['sell_price']),
                format_price(current_price),
                format_price(tracked_trade['highest_price']),
                f"{((tracked_trade['highest_price'] - tracked_trade['buy_price']) / tracked_trade['buy_price']) * 100:.2f}%"
            ])

        tracked_sold_headers = ["Token Name", "Token Address", "Buy Price USD", "Sell Price USD", "Current Price USD",
                                "Highest Price USD", "% Apprec to High"]
        print("\n=== Tracked Sold Tokens (All Sold Tokens) ===")
        print(tabulate(tracked_sold_data, headers=tracked_sold_headers, tablefmt="pretty",
                       maxcolwidths=[20, 45, 15, 15, 15, 15, 15]))

def calculate_profit_loss(trades):
    """Calculate total profit/loss from completed trades and unrealized PnL for active trades"""
    total_profit_loss = sum(trade["profit_loss"] for trade in trades["sells"])
    current_value = sum(trade["current_price"] * trade["quantity"] for trade in trades["buys"])
    capital_spent = sum(trade["buy_price"] * trade["quantity"] for trade in trades["buys"])
    unrealized_pnl = current_value - capital_spent if trades["buys"] else 0.0
    return total_profit_loss, current_value, unrealized_pnl

def calculate_avg_percent_increase(trades, processed_tokens):
    """Calculate the average percentage increase to the highest price for all bought tokens"""
    if not trades["buys"] and not trades["sells"] and not trades.get("tracked_sold", []):
        return 0.0, []

    percent_increases = []
    tokens_to_remove = []

    def process_trade(trade, is_active=True):
        if trade["buy_price"] <= 0:
            return None, None
        token_address = trade["token_address"]
        token = next((t for t in processed_tokens if t["token_address"] == token_address), None)
        current_price = token["price_usd"] if token and token["price_usd"] != "N/A" and isinstance(token["price_usd"], (int, float)) else trade["highest_price"]
        if current_price <= trade["buy_price"] * 0.8:
            return None, token_address
        percent_increase = ((trade["highest_price"] - trade["buy_price"]) / trade["buy_price"]) * 100
        return percent_increase, None

    for trade in trades["buys"][:]:
        percent_increase, to_remove = process_trade(trade, is_active=True)
        if percent_increase is not None:
            percent_increases.append(percent_increase)
        elif to_remove:
            tokens_to_remove.append(to_remove)
            trades["buys"].remove(trade)

    for trade in trades["sells"][:]:
        percent_increase, to_remove = process_trade(trade, is_active=False)
        if percent_increase is not None:
            percent_increases.append(percent_increase)
        elif to_remove:
            tokens_to_remove.append(to_remove)

    for tracked_trade in trades.get("tracked_sold", [])[:]:
        percent_increase, to_remove = process_trade(tracked_trade, is_active=False)
        if percent_increase is not None:
            percent_increases.append(percent_increase)
        elif to_remove:
            tokens_to_remove.append(to_remove)
            trades["tracked_sold"].remove(tracked_trade)

    avg_increase = sum(percent_increases) / len(percent_increases) if percent_increases else 0.0
    return avg_increase, tokens_to_remove

def load_trading_data():
    """Load existing trading data from JSON file, or initialize if not present"""
    try:
        with open("trading_data.json", "r") as f:
            data = json.load(f)
            for trade in data["buys"]:
                if "highest_price" not in trade:
                    trade["highest_price"] = trade["buy_price"]
                if "buy_time" not in trade:
                    trade["buy_time"] = int(time.time())
            for trade in data["sells"]:
                if "highest_price" not in trade:
                    trade["highest_price"] = trade["buy_price"]
                if "buy_time" not in trade:
                    trade["buy_time"] = int(time.time())
                if "sell_time" not in trade:
                    trade["sell_time"] = int(time.time())
                if "quantity" not in trade:
                    trade["quantity"] = 1.0
            for trade in data.get("tracked_sold", []):
                if "highest_price" not in trade:
                    trade["highest_price"] = trade["sell_price"]
            return data
    except FileNotFoundError:
        return {
            "capital": INITIAL_CAPITAL,
            "buys": [],
            "sells": [],
            "tracked_sold": [],
            "known_tokens": []
        }

def save_trading_data(trading_data):
    """Save trading data to JSON file"""
    with open("trading_data.json", "w") as f:
        json.dump(trading_data, f, indent=2)

def main():
    global FIRST_RUN
    trading_data = load_trading_data()
    current_capital = trading_data["capital"]
    known_tokens = set(trading_data["known_tokens"])
    start_time = time.time()
    iteration = 0

    while current_capital > 0:
        iteration += 1
        runtime_minutes = (time.time() - start_time) / 60
        print(f"\n=== Running at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Iteration {iteration}, Runtime: {runtime_minutes:.2f} minutes ===")

        boosted_tokens = get_boosted_tokens()
        if not boosted_tokens:
            print("Failed to fetch boosted tokens. Retrying in 60 seconds...")
            time.sleep(60)
            continue

        processed_tokens = process_token_data(boosted_tokens)
        sorted_tokens = sorted(processed_tokens, key=lambda x: x["age_minutes"])

        if FIRST_RUN:
            print("Skipping buys on first run...")
            FIRST_RUN = False
        else:
            new_tokens = [token for token in processed_tokens if
                          token["token_address"] not in known_tokens and token["price_usd"] != "N/A" and
                          (token["liquidity_usd"] == 0 or token["liquidity_usd"] >= 10000)]
            for token in new_tokens:
                buy_amount = current_capital * BUY_PERCENTAGE
                price_usd = token["price_usd"]
                if isinstance(price_usd, (int, float)) and price_usd > 0:
                    quantity = buy_amount / price_usd
                    if quantity > 0:
                        print(f"Buying {token['token_name']} at ${format_price(price_usd)} with ${buy_amount:,.2f} ({quantity:.2f} units, CA: {token['token_address']})")
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
                    else:
                        print(f"Skipping {token['token_name']}: Insufficient capital for any units (${format_price(price_usd)} each, CA: {token['token_address']})")
                else:
                    print(f"Skipping {token['token_name']}: Price USD is N/A or invalid (CA: {token['token_address']})")

        current_time = int(time.time())
        for trade in trading_data["buys"][:]:
            token_address = trade["token_address"]
            token = next((t for t in processed_tokens if t["token_address"] == token_address), None)
            if token and token["price_usd"] != "N/A" and isinstance(token["price_usd"], (int, float)):
                current_price = token["price_usd"]
                trade["current_price"] = current_price
                trade["highest_price"] = max(trade["highest_price"], current_price)

                if current_price >= trade["buy_price"] * SELL_THRESHOLD:
                    sell_value = current_price * trade["quantity"]
                    profit_loss = sell_value - (trade["buy_price"] * trade["quantity"])
                    print(f"Selling {trade['token_name']} at ${format_price(current_price)} (Bought at ${format_price(trade['buy_price'])}, Profit/Loss: ${profit_loss:.2f}, CA: {trade['token_address']}) - Reason: Reached profit target")
                    current_capital += sell_value
                    trading_data["sells"].append({
                        "token_name": trade["token_name"],
                        "token_address": trade["token_address"],
                        "buy_price": trade["buy_price"],
                        "sell_price": current_price,
                        "profit_loss": profit_loss,
                        "highest_price": trade["highest_price"],
                        "buy_time": trade["buy_time"],
                        "sell_time": int(time.time()),
                        "quantity": trade["quantity"]
                    })
                    trading_data["tracked_sold"].append({
                        "token_name": trade["token_name"],
                        "token_address": trade["token_address"],
                        "buy_price": trade["buy_price"],
                        "sell_price": current_price,
                        "highest_price": current_price
                    })
                    trading_data["buys"].remove(trade)

                elif current_price <= trade["buy_price"] * STOP_LOSS_THRESHOLD:
                    sell_value = current_price * trade["quantity"]
                    profit_loss = sell_value - (trade["buy_price"] * trade["quantity"])
                    print(f"Stop Loss: Selling {trade['token_name']} at ${format_price(current_price)} (Bought at ${format_price(trade['buy_price'])}, Profit/Loss: ${profit_loss:.2f}, CA: {trade['token_address']}) - Reason: Hit stop loss")
                    current_capital += sell_value
                    trading_data["sells"].append({
                        "token_name": trade["token_name"],
                        "token_address": trade["token_address"],
                        "buy_price": trade["buy_price"],
                        "sell_price": current_price,
                        "profit_loss": profit_loss,
                        "highest_price": trade["highest_price"],
                        "buy_time": trade["buy_time"],
                        "sell_time": int(time.time()),
                        "quantity": trade["quantity"]
                    })
                    trading_data["tracked_sold"].append({
                        "token_name": trade["token_name"],
                        "token_address": trade["token_address"],
                        "buy_price": trade["buy_price"],
                        "sell_price": current_price,
                        "highest_price": current_price
                    })
                    trading_data["buys"].remove(trade)

                elif token["price_change_5m"] == 0:
                    sell_value = current_price * trade["quantity"]
                    profit_loss = sell_value - (trade["buy_price"] * trade["quantity"])
                    print(f"Selling {trade['token_name']} at ${format_price(current_price)} (Bought at ${format_price(trade['buy_price'])}, Profit/Loss: ${profit_loss:.2f}, CA: {trade['token_address']}) - Reason: No buy activity in last 5 minutes")
                    current_capital += sell_value
                    trading_data["sells"].append({
                        "token_name": trade["token_name"],
                        "token_address": trade["token_address"],
                        "buy_price": trade["buy_price"],
                        "sell_price": current_price,
                        "profit_loss": profit_loss,
                        "highest_price": trade["highest_price"],
                        "buy_time": trade["buy_time"],
                        "sell_time": int(time.time()),
                        "quantity": trade["quantity"]
                    })
                    trading_data["tracked_sold"].append({
                        "token_name": trade["token_name"],
                        "token_address": trade["token_address"],
                        "buy_price": trade["buy_price"],
                        "sell_price": current_price,
                        "highest_price": current_price
                    })
                    trading_data["buys"].remove(trade)

        for tracked_trade in trading_data.get("tracked_sold", [])[:]:
            token_address = tracked_trade["token_address"]
            token = next((t for t in processed_tokens if t["token_address"] == token_address), None)
            if token and token["price_usd"] != "N/A" and isinstance(token["price_usd"], (int, float)):
                current_price = token["price_usd"]
                tracked_trade["highest_price"] = max(tracked_trade["highest_price"], current_price)
                if current_price <= tracked_trade["buy_price"] * 0.8:
                    print(f"Stopping tracking {tracked_trade['token_name']} (CA: {tracked_trade['token_address']}) - Price dropped 80% from buy price (${tracked_trade['buy_price']} to ${current_price})")
                    trading_data["tracked_sold"].remove(tracked_trade)

        avg_percent_increase, tokens_to_remove = calculate_avg_percent_increase(trading_data, processed_tokens)

        for token_address in tokens_to_remove:
            if any(trade["token_address"] == token_address for trade in trading_data["buys"]):
                trading_data["buys"] = [trade for trade in trading_data["buys"] if trade["token_address"] != token_address]
            if any(trade["token_address"] == token_address for trade in trading_data["sells"]):
                trading_data["sells"] = [trade for trade in trading_data["sells"] if trade["token_address"] != token_address]

        known_tokens.update(token["token_address"] for token in processed_tokens if token["token_address"] not in tokens_to_remove)
        trading_data["known_tokens"] = list(known_tokens)
        trading_data["capital"] = current_capital

        display_tokens(sorted_tokens, current_capital, trading_data, iteration, runtime_minutes)

        total_profit_loss, current_value, unrealized_pnl = calculate_profit_loss(trading_data)
        print(f"\n=== Trading Summary - Iteration {iteration} ===")
        print(f"Total Capital: ${current_capital:,.2f}")
        print(f"Current Holdings Value: ${current_value:,.2f}")
        print(f"Total Realized Profit/Loss: ${total_profit_loss:,.2f}")
        print(f"Unrealized Profit/Loss: ${unrealized_pnl:,.2f}")
        print(f"Number of Active Trades: {len(trading_data['buys'])}")
        print(f"Number of Completed Trades: {len(trading_data['sells'])}")
        print(f"Number of Tracked Sold Tokens: {len(trading_data.get('tracked_sold', []))}")
        print(f"Average % Increase to Highest Price: ${avg_percent_increase:.2f}%")

        save_trading_data(trading_data)
        time.sleep(60)

    print("\n=== All capital lost. Simulation ended. ===")
    total_profit_loss, current_value, unrealized_pnl = calculate_profit_loss(trading_data)
    avg_percent_increase, _ = calculate_avg_percent_increase(trading_data, processed_tokens)
    runtime_minutes = (time.time() - start_time) / 60
    print(f"Final Report - Iteration {iteration}, Runtime: {runtime_minutes:.2f} minutes:")
    print(f"Final Capital: ${current_capital:,.2f}")
    print(f"Current Holdings Value: ${current_value:,.2f}")
    print(f"Total Realized Profit/Loss: ${total_profit_loss:,.2f}")
    print(f"Unrealized Profit/Loss: ${unrealized_pnl:,.2f}")
    print(f"Number of Active Trades: {len(trading_data['buys'])}")
    print(f"Number of Completed Trades: {len(trading_data['sells'])}")
    print(f"Number of Tracked Sold Tokens: {len(trading_data.get('tracked_sold', []))}")
    print(f"Average % Increase to Highest Price: ${avg_percent_increase:.2f}%")

if __name__ == "__main__":
    main()