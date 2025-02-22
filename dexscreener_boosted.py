import requests
import json
from datetime import datetime
import time
from tabulate import tabulate

# Constants
API_BASE_URL = "https://api.dexscreener.com"
HEADERS = {
    "User-Agent": "BoostedTokenAnalyzer/1.0",
    "Accept": "application/json"
}
INITIAL_CAPITAL = 10000.0
BUY_PERCENTAGE = 0.1
SELL_THRESHOLD = 1.4
STOP_LOSS_THRESHOLD = 0.7
CHAIN_ID = "solana"
FIRST_RUN = True


# API Functions
def get_boosted_tokens():
    """Fetch the latest boosted tokens from the DEX Screener API."""
    try:
        response = requests.get(f"{API_BASE_URL}/token-boosts/latest/v1", headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Ensure data is a list; if it's a single object, wrap it
        if isinstance(data, dict):
            return [data]
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching boosted tokens: {e}")
        return None


def get_token_pairs(chain_id, token_address):
    """Fetch pair data for a specific token on Solana."""
    try:
        response = requests.get(f"{API_BASE_URL}/token-pairs/v1/{chain_id}/{token_address}", headers=HEADERS,
                                timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching token pairs for {token_address}: {e}")
        return None


def get_pair_data(chain_id, pair_address):
    """Fetch detailed pair data for a specific pair."""
    try:
        response = requests.get(f"{API_BASE_URL}/latest/dex/pairs/{chain_id}/{pair_address}", headers=HEADERS,
                                timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["pairs"][0] if "pairs" in data and data["pairs"] else None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching pair data for {pair_address}: {e}")
        return None


# Helper Functions
def format_price(price):
    """Format a price to show up to 10 decimal places, removing trailing zeros and decimal if no significant digits."""
    if isinstance(price, (int, float)):
        formatted = f"{price:.10f}".rstrip('0').rstrip('.')
        return formatted if formatted else "0"
    else:
        return "N/A"


def process_token_data(token_data):
    """Process and filter token data for Solana tokens, calculating age and extracting necessary fields."""
    processed_tokens = []
    current_time = int(time.time())
    for token in token_data:
        if token.get("chainId") != CHAIN_ID:
            continue
        token_address = token["tokenAddress"]
        boost_amount = float(token.get("amount", "0"))
        description = token.get("description", "")
        token_name = description.split()[0] if description else "Unknown"
        pairs = get_token_pairs(CHAIN_ID, token_address)
        if pairs:
            for pair in pairs:
                pair_address = pair["pairAddress"]
                pair_created_at = pair.get("pairCreatedAt", 0) / 1000  # Convert milliseconds to seconds
                age_minutes = (current_time - pair_created_at) / 60 if pair_created_at else 0
                price_usd = float(pair.get("priceUsd", "0"))
                liquidity_usd = float(pair.get("liquidity", {}).get("usd", "0"))
                fdv = float(pair.get("fdv", "0"))
                market_cap = float(pair.get("marketCap", "0"))
                volume_24h = float(pair.get("volume", {}).get("h24", "0"))
                txns_24h = int(pair.get("txns", {}).get("h24", {}).get("buys", 0)) + \
                           int(pair.get("txns", {}).get("h24", {}).get("sells", 0))
                buys_24h = int(pair.get("txns", {}).get("h24", {}).get("buys", 0))
                sells_24h = int(pair.get("txns", {}).get("h24", {}).get("sells", 0))
                price_change_5m = float(pair.get("priceChange", {}).get("m5", "0"))
                price_change_1h = float(pair.get("priceChange", {}).get("h1", "0"))
                price_change_6h = float(pair.get("priceChange", {}).get("h6", "0"))
                price_change_24h = float(pair.get("priceChange", {}).get("h24", "0"))
                processed_tokens.append({
                    "age_minutes": age_minutes,
                    "token_address": token_address,
                    "chain_id": CHAIN_ID,
                    "boost_amount": boost_amount,
                    "pair_address": pair_address,
                    "token_name": token_name,
                    "price_usd": price_usd,
                    "liquidity_usd": liquidity_usd,
                    "fdv": fdv,
                    "market_cap": market_cap,
                    "volume_24h": volume_24h,
                    "txns_24h": txns_24h,
                    "buys_24h": buys_24h,
                    "sells_24h": sells_24h,
                    "buy_vol_24h": "N/A",  # API does not provide separate buy/sell volumes
                    "sell_vol_24h": "N/A",
                    "price_change_5m": price_change_5m,
                    "price_change_1h": price_change_1h,
                    "price_change_6h": price_change_6h,
                    "price_change_24h": price_change_24h,
                })
    processed_tokens.sort(key=lambda x: x["age_minutes"])  # Sort by age (newest first, smallest age_minutes)
    return processed_tokens


def display_tokens(tokens, current_capital, trades, iteration, runtime_minutes):
    """Display token and trade information in tabulated format."""
    pair_data_map = {pair["pair_address"]: pair for pair in tokens}

    # Boosted Tokens Table (Top 10)
    boosted_tokens_table = []
    for token in tokens[:10]:
        boosted_tokens_table.append([
            f"{token['age_minutes']:.1f}",
            token['token_name'],
            token['token_address'],
            token['pair_address'],
            f"{token['boost_amount']:.2f}",
            format_price(token['price_usd']),
            f"{token['liquidity_usd']:,.2f}",
            f"{token['fdv']:,.2f}" if token['fdv'] > 0 else "N/A",
            f"{token['market_cap']:,.2f}" if token['market_cap'] > 0 else "N/A",
            token['txns_24h'],
            token['buys_24h'],
            token['sells_24h'],
            token['buy_vol_24h'],
            token['sell_vol_24h'],
            f"{token['price_change_5m']:.2f}%",
            f"{token['price_change_1h']:.2f}%",
            f"{token['price_change_6h']:.2f}%",
            f"{token['price_change_24h']:.2f}%",
            token['chain_id']
        ])
    print("\nBoosted Tokens (Top 10):")
    print(tabulate(boosted_tokens_table, headers=[
        "Age (m)", "Name", "Token Address", "Pair Address", "Boost", "Price USD",
        "Liquidity USD", "FDV", "Market Cap", "Txns (24h)", "Buys (24h)", "Sells (24h)",
        "Buy Vol (24h)", "Sell Vol (24h)", "Price Chg 5m", "Price Chg 1h",
        "Price Chg 6h", "Price Chg 24h", "Chain"
    ], tablefmt="pretty", maxcolwidths=[10, 20, 45, 45, 10, 12, 15, 15, 15, 10, 10, 10, 15, 15, 10, 10, 10, 10, 10]))

    print(f"\nCurrent Capital: ${current_capital:,.2f}")

    # Active Trades Table
    active_trades_table = []
    for trade in trades["buys"]:
        current_price = trade.get("current_price", 0)
        buy_price = trade["buy_price"]
        quantity = trade["quantity"]
        capital_spent = trade["capital_spent"]
        current_value = current_price * quantity
        percent_change = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
        pnl = current_value - capital_spent
        highest_price = trade["highest_price"]
        hold_age_minutes = (time.time() - trade["buy_time"]) / 60
        percent_diff_to_high = ((highest_price - current_price) / highest_price * 100) if highest_price > 0 else 0
        last_buy_minutes = 0.0 if trade.get("price_change_5m", 0) != 0 else 5.0
        active_trades_table.append([
            trade['token_name'],
            trade['token_address'],
            format_price(buy_price),
            format_price(current_price),
            f"{quantity:.2f}",
            f"{capital_spent:,.2f}",
            f"{current_value:,.2f}",
            f"{percent_change:.2f}%",
            f"{pnl:,.2f}",
            format_price(highest_price),
            f"{hold_age_minutes:.1f}",
            f"{percent_diff_to_high:.2f}%",
            f"{last_buy_minutes:.1f}"
        ])
    print("\nActive Trades:")
    print(tabulate(active_trades_table, headers=[
        "Token Name", "Token Address", "Buy Price USD", "Current Price USD", "Quantity",
        "Capital Spent", "Current Value", "% Change", "PnL", "Highest Price USD",
        "Hold Age (m)", "% Diff to High", "Last Buy (m)"
    ], tablefmt="pretty", maxcolwidths=[20, 45, 15, 15, 10, 15, 15, 10, 15, 15, 10, 15, 10]))

    # Completed Trades Table
    completed_trades_table = []
    for trade in trades["sells"]:
        buy_price = trade["buy_price"]
        sell_price = trade["sell_price"]
        capital_spent = trade["capital_spent"]
        sell_value = trade["sell_value"]
        pnl = trade["profit_loss"]
        highest_price = trade["highest_price"]
        hold_age_minutes = (trade["sell_time"] - trade["buy_time"]) / 60
        completed_trades_table.append([
            trade['token_name'],
            trade['token_address'],
            format_price(buy_price),
            format_price(sell_price),
            f"{capital_spent:,.2f}",
            f"{sell_value:,.2f}",
            f"{pnl:,.2f}",
            format_price(highest_price),
            f"{hold_age_minutes:.1f}"
        ])
    print("\nCompleted Trades:")
    print(tabulate(completed_trades_table, headers=[
        "Token Name", "Token Address", "Buy Price USD", "Sell Price USD", "Capital Spent",
        "Sell Value", "PnL", "Highest Price USD", "Hold Age (m)"
    ], tablefmt="pretty", maxcolwidths=[20, 45, 15, 15, 15, 15, 15, 15, 15]))

    # Tracked Sold Tokens Table
    tracked_sold_table = []
    for trade in trades["tracked_sold"]:
        current_price = pair_data_map[trade["pair_address"]]["price_usd"] if trade[
                                                                                 "pair_address"] in pair_data_map else "N/A"
        tracked_sold_table.append([
            trade['token_name'],
            trade['token_address'],
            format_price(trade["buy_price"]),
            format_price(trade["sell_price"]),
            format_price(current_price) if current_price != "N/A" else "N/A",
            format_price(trade["highest_price"]),
            f"{((trade['highest_price'] - trade['buy_price']) / trade['buy_price'] * 100):.2f}%" if trade[
                                                                                                        'buy_price'] > 0 else "N/A"
        ])
    print("\nTracked Sold Tokens:")
    print(tabulate(tracked_sold_table, headers=[
        "Token Name", "Token Address", "Buy Price USD", "Sell Price USD", "Current Price USD",
        "Highest Price USD", "% Apprec to High"
    ], tablefmt="pretty", maxcolwidths=[20, 45, 15, 15, 15, 15, 15]))


def calculate_profit_loss(trades):
    """Calculate total realized profit/loss and unrealized PnL for active trades."""
    realized_pnl = sum(trade["profit_loss"] for trade in trades["sells"])
    current_holdings_value = sum(trade["current_price"] * trade["quantity"] for trade in trades["buys"])
    capital_spent_on_buys = sum(trade["capital_spent"] for trade in trades["buys"])
    unrealized_pnl = current_holdings_value - capital_spent_on_buys
    return realized_pnl, current_holdings_value, unrealized_pnl


def calculate_avg_percent_increase(trades, processed_tokens):
    """Calculate the average percentage increase to the highest price for all bought tokens."""
    pair_data_map = {pair["pair_address"]: pair for pair in processed_tokens}
    percent_increases = []
    tokens_to_remove = []
    for trade in trades["buys"] + trades["sells"] + trades["tracked_sold"]:
        buy_price = trade["buy_price"]
        if buy_price <= 0:
            continue
        current_price = pair_data_map.get(trade["pair_address"], {}).get("price_usd", trade.get("current_price",
                                                                                                trade["highest_price"]))
        if current_price <= buy_price * 0.8:
            tokens_to_remove.append(trade["token_address"])
            continue
        highest_price = trade["highest_price"]
        percent_increase = ((highest_price - buy_price) / buy_price) * 100
        percent_increases.append(percent_increase)
    avg_percent_increase = sum(percent_increases) / len(percent_increases) if percent_increases else 0
    return avg_percent_increase, tokens_to_remove


def load_trading_data():
    """Load or initialize trading data from trading_data.json."""
    try:
        with open("trading_data.json", "r") as f:
            data = json.load(f)
            # Ensure all necessary fields are present
            for trade_list in ["buys", "sells", "tracked_sold"]:
                for trade in data.get(trade_list, []):
                    trade["highest_price"] = trade.get("highest_price", trade.get("buy_price", 0))
                    trade["buy_time"] = trade.get("buy_time", time.time())
                    if trade_list == "sells":
                        trade["sell_time"] = trade.get("sell_time", time.time())
                        trade["quantity"] = trade.get("quantity", 1.0)
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
    """Save trading data to trading_data.json."""
    with open("trading_data.json", "w") as f:
        json.dump(trading_data, f, indent=4)


# Main Logic
def main():
    """Main function to run the trading simulation."""
    global FIRST_RUN
    trading_data = load_trading_data()
    current_capital = trading_data["capital"]
    known_tokens = set(trading_data["known_tokens"])
    iteration = 0
    start_time = time.time()

    while current_capital > 0:
        iteration += 1
        runtime_minutes = (time.time() - start_time) / 60
        print(
            f"\nIteration {iteration} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Runtime: {runtime_minutes:.1f} minutes)")

        token_data = get_boosted_tokens()
        if token_data is None:
            time.sleep(60)
            continue
        processed_tokens = process_token_data(token_data)
        pair_data_map = {pair["pair_address"]: pair for pair in processed_tokens}

        if not FIRST_RUN:
            # Simulate buying new tokens
            for token in processed_tokens:
                token_address = token["token_address"]
                pair_address = token["pair_address"]
                price_usd = token["price_usd"]
                liquidity_usd = token["liquidity_usd"]
                if token_address not in known_tokens and price_usd > 0 and (
                        liquidity_usd == 0 or liquidity_usd >= 10000):
                    capital_to_spend = current_capital * BUY_PERCENTAGE
                    if capital_to_spend > current_capital:
                        print(f"Insufficient capital to buy {token['token_name']}")
                        continue
                    quantity = capital_to_spend / price_usd
                    if quantity > 0:
                        current_capital -= capital_to_spend
                        buy_time = time.time()
                        trading_data["buys"].append({
                            "token_address": token_address,
                            "pair_address": pair_address,
                            "token_name": token["token_name"],
                            "buy_price": price_usd,
                            "quantity": quantity,
                            "capital_spent": capital_to_spend,
                            "buy_time": buy_time,
                            "highest_price": price_usd,
                            "current_price": price_usd,
                            "price_change_5m": token["price_change_5m"]
                        })
                        known_tokens.add(token_address)
                        print(
                            f"Bought {quantity:.2f} of {token['token_name']} at ${price_usd:.10f} for ${capital_to_spend:,.2f}")
        FIRST_RUN = False

        # Update and check sell conditions for active trades
        for trade in trading_data["buys"][:]:
            pair_address = trade["pair_address"]
            if pair_address in pair_data_map:
                current_price = pair_data_map[pair_address]["price_usd"]
                price_change_5m = pair_data_map[pair_address]["price_change_5m"]
                trade["current_price"] = current_price
                trade["price_change_5m"] = price_change_5m
                trade["highest_price"] = max(trade["highest_price"], current_price)
            else:
                # Keep last known values
                pass

            buy_price = trade["buy_price"]
            current_price = trade["current_price"]
            price_change_5m = trade["price_change_5m"]

            sell_reason = None
            if current_price >= buy_price * SELL_THRESHOLD:
                sell_reason = "Reached profit target"
            elif current_price <= buy_price * STOP_LOSS_THRESHOLD:
                sell_reason = "Hit stop loss"
            elif price_change_5m == 0:
                sell_reason = "No buy activity in last 5 minutes"

            if sell_reason:
                sell_value = current_price * trade["quantity"]
                profit_loss = sell_value - trade["capital_spent"]
                current_capital += sell_value
                trading_data["sells"].append({
                    "token_address": trade["token_address"],
                    "pair_address": trade["pair_address"],
                    "token_name": trade["token_name"],
                    "buy_price": buy_price,
                    "sell_price": current_price,
                    "quantity": trade["quantity"],
                    "capital_spent": trade["capital_spent"],
                    "sell_value": sell_value,
                    "profit_loss": profit_loss,
                    "buy_time": trade["buy_time"],
                    "sell_time": time.time(),
                    "highest_price": trade["highest_price"]
                })
                trading_data["tracked_sold"].append({
                    "token_address": trade["token_address"],
                    "pair_address": trade["pair_address"],
                    "token_name": trade["token_name"],
                    "buy_price": buy_price,
                    "sell_price": current_price,
                    "highest_price": trade["highest_price"]
                })
                trading_data["buys"].remove(trade)
                print(
                    f"Sold {trade['token_name']} at ${current_price:.10f} for ${sell_value:,.2f} (Reason: {sell_reason})")

        # Update tracked sold tokens
        for trade in trading_data["tracked_sold"][:]:
            pair_address = trade["pair_address"]
            if pair_address in pair_data_map:
                current_price = pair_data_map[pair_address]["price_usd"]
                trade["highest_price"] = max(trade["highest_price"], current_price)
                if current_price <= trade["buy_price"] * 0.8:
                    trading_data["tracked_sold"].remove(trade)
                    print(f"Stopped tracking {trade['token_name']} (Price dropped 80%)")

        # Calculate average % increase and tokens to remove
        avg_percent_increase, tokens_to_remove = calculate_avg_percent_increase(trading_data, processed_tokens)
        for token_address in tokens_to_remove:
            known_tokens.discard(token_address)

        # Display tokens and trades
        display_tokens(processed_tokens, current_capital, trading_data, iteration, runtime_minutes)

        # Report trading summary
        realized_pnl, current_holdings_value, unrealized_pnl = calculate_profit_loss(trading_data)
        print(f"\nTrading Summary:")
        print(f"  Current Capital: ${current_capital:,.2f}")
        print(f"  Holdings Value: ${current_holdings_value:,.2f}")
        print(f"  Realized P/L: ${realized_pnl:,.2f}")
        print(f"  Unrealized P/L: ${unrealized_pnl:,.2f}")
        print(f"  Active Trades: {len(trading_data['buys'])}")
        print(f"  Completed Trades: {len(trading_data['sells'])}")
        print(f"  Average % Increase to High: {avg_percent_increase:.2f}%")

        # Save trading data
        trading_data["capital"] = current_capital
        trading_data["known_tokens"] = list(known_tokens)
        save_trading_data(trading_data)

        time.sleep(60)

    # Final report
    print("\nCapital depleted. Stopping simulation.")
    realized_pnl, current_holdings_value, unrealized_pnl = calculate_profit_loss(trading_data)
    print(f"Final Trading Summary:")
    print(f"  Final Capital: ${current_capital:,.2f}")
    print(f"  Holdings Value: ${current_holdings_value:,.2f}")
    print(f"  Total Realized P/L: ${realized_pnl:,.2f}")
    print(f"  Total Unrealized P/L: ${unrealized_pnl:,.2f}")
    print(f"  Total Trades: {len(trading_data['sells'])}")


if __name__ == "__main__":
    main()