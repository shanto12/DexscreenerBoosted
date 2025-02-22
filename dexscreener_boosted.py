import requests
import json
from datetime import datetime
import time
from tabulate import tabulate
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("trading_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "https://api.dexscreener.com"
HEADERS = {"User-Agent": "BoostedTokenAnalyzer/1.0", "Accept": "application/json"}
INITIAL_CAPITAL = 10000.0
BUY_PERCENTAGE = 0.2  # 20% of available capital
SELL_THRESHOLD = 1.5  # 50% profit
PARTIAL_SELL_THRESHOLD = 1.25  # 25% profit for partial sell
STOP_LOSS_THRESHOLD = 0.7  # 30% stop loss
TRAILING_STOP_PERCENT = 0.1  # 10% trailing stop
CHAIN_ID = "solana"
FIRST_RUN = True
MAX_BUYS_PER_ITERATION = 5
NO_CHANGE_SELL_MINUTES = 15

# Required fields for trades
REQUIRED_FIELDS = {
    "buys": ["token_address", "pair_address", "token_name", "buy_price", "quantity", "capital_spent", "buy_time",
             "highest_price"],
    "sells": ["token_address", "pair_address", "token_name", "buy_price", "sell_price", "quantity", "capital_spent",
              "sell_value", "profit_loss", "buy_time", "sell_time", "highest_price"],
    "tracked_sold": ["token_address", "pair_address", "token_name", "buy_price", "sell_price", "highest_price"]
}

# Cache for pair data
pair_cache = {}


# API Functions
def get_boosted_tokens():
    try:
        response = requests.get(f"{API_BASE_URL}/token-boosts/latest/v1", headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Fetched boosted tokens: {len(data) if isinstance(data, list) else 1} entries")
        return data if isinstance(data, list) else [data]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching boosted tokens: {e}")
        return None


def get_token_pairs(token_address):
    try:
        response = requests.get(
            f"{API_BASE_URL}/latest/dex/tokens/{token_address}",
            headers=HEADERS,  # Assuming HEADERS is defined globally
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        pairs = [pair for pair in data.get("pairs", []) if pair["chainId"] == CHAIN_ID]
        logger.info(f"Fetched {len(pairs)} pairs for token {token_address}")
        return pairs
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching token pairs for {token_address}: {e}")
        return []



# Helper Functions
def format_price(price):
    if isinstance(price, (int, float)) and price > 0:
        formatted = f"{price:.10f}".rstrip('0').rstrip('.')
        return formatted if formatted else "0"
    return "N/A"


def aggregate_boosts(token_data):
    aggregated = {}
    for token in token_data:
        if token.get("chainId") != CHAIN_ID:  # Assuming CHAIN_ID is defined globally
            continue
        token_address = token["tokenAddress"]
        boost_amount = float(token.get("amount", "0"))
        description = token.get("description", "")
        if token_address not in aggregated:
            aggregated[token_address] = {
                "token_address": token_address,
                "boost_amount": 0,
                "description": description
            }
        aggregated[token_address]["boost_amount"] += boost_amount
    logger.info(f"Aggregated {len(aggregated)} unique tokens")
    return list(aggregated.values())


def process_token_data(aggregated_tokens):
    processed_tokens = []
    current_time = int(time.time())
    for agg_token in aggregated_tokens:
        token_address = agg_token["token_address"]
        boost_amount = agg_token["boost_amount"]
        token_name = agg_token["description"].split()[0] if agg_token["description"] else "Unknown"
        pairs = get_token_pairs(token_address)
        if pairs:
            for pair in pairs:
                pair_address = pair["pairAddress"]
                pair_created_at = pair.get("pairCreatedAt", 0) / 1000  # Convert ms to seconds
                age_minutes = (current_time - pair_created_at) / 60 if pair_created_at else 0
                price_usd = float(pair.get("priceUsd", "0"))
                liquidity_usd = float(pair.get("liquidity", {}).get("usd", "0"))
                fdv = float(pair.get("fdv", "0"))
                market_cap = float(pair.get("marketCap", "0"))
                txns_24h = int(pair.get("txns", {}).get("h24", {}).get("buys", 0)) + \
                          int(pair.get("txns", {}).get("h24", {}).get("sells", 0))
                price_change_5m = float(pair.get("priceChange", {}).get("m5", "0"))
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
                    "txns_24h": txns_24h,
                    "price_change_5m": price_change_5m,
                })
    processed_tokens.sort(key=lambda x: x["age_minutes"])
    logger.info(f"Processed {len(processed_tokens)} unique token-pair combinations")
    return processed_tokens


def display_tokens(tokens, current_capital, trades, iteration, runtime_minutes):
    pair_data_map = {pair["pair_address"]: pair for pair in tokens}

    # Boosted Tokens Table
    boosted_tokens_table = []
    for token in tokens[:10]:
        market_cap_display = f"{token['market_cap']:,.2f}" if token['market_cap'] > 0 else "N/A"
        fdv_display = f"{token['fdv']:,.2f}" if token['fdv'] > 0 else "N/A"
        boosted_tokens_table.append([
            f"{token['age_minutes']:.1f}",
            token['token_name'],
            token['token_address'],
            token['pair_address'],
            f"{token['boost_amount']:.2f}",
            format_price(token['price_usd']),
            f"{token['liquidity_usd']:,.2f}",
            fdv_display,
            market_cap_display,
            token['txns_24h'],
            f"{token['price_change_5m']:.2f}%",
            token['chain_id']
        ])
    print(f"\n=== Iteration {iteration} Boosted Tokens (Top 10) ===")
    print(tabulate(boosted_tokens_table, headers=[
        "Age (m)", "Name", "Token Address", "Pair Address", "Total Boost", "Price USD",
        "Liquidity USD", "FDV", "Market Cap", "Txns (24h)", "Price Chg 5m", "Chain"
    ], tablefmt="pretty"))
    print(f"\n=== Current Capital ===\n${current_capital:,.2f}")

    # Active Trades Table
    active_trades_table = []
    for trade in trades["buys"]:
        current_price = pair_data_map.get(trade["pair_address"], {}).get("price_usd", trade["buy_price"])
        trade["current_price"] = current_price
        trade["highest_price"] = max(trade["highest_price"], current_price)
        buy_price = trade["buy_price"]
        quantity = trade["quantity"]
        capital_spent = trade["capital_spent"]
        current_value = current_price * quantity
        percent_change = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
        pnl = current_value - capital_spent
        hold_age_minutes = (time.time() - trade["buy_time"]) / 60
        percent_diff_to_high = ((trade["highest_price"] - current_price) / trade["highest_price"] * 100) if trade[
                                                                                                                "highest_price"] > 0 else 0
        active_trades_table.append([
            trade['token_name'],
            trade['token_address'],
            format_price(buy_price),
            format_price(current_price),
            f"{quantity:,.2f}",
            f"{capital_spent:,.2f}",
            f"{current_value:,.2f}",
            f"{percent_change:.2f}%",
            f"{pnl:,.2f}",
            format_price(trade["highest_price"]),
            f"{hold_age_minutes:.1f}",
            f"{percent_diff_to_high:.2f}%"
        ])
    print("\n=== Active Trades ===")
    print(tabulate(active_trades_table, headers=[
        "Token Name", "Token Address", "Buy Price USD", "Current Price USD", "Quantity",
        "Capital Spent", "Current Value", "% Change", "PnL", "Highest Price USD",
        "Hold Age (m)", "% Diff to High"
    ], tablefmt="pretty") if active_trades_table else "No active trades.")

    # Completed Trades Table
    completed_trades_table = []
    for trade in trades["sells"]:
        completed_trades_table.append([
            trade['token_name'],
            trade['token_address'],
            format_price(trade["buy_price"]),
            format_price(trade["sell_price"]),
            f"{trade['quantity']:,.2f}",
            f"{trade['capital_spent']:,.2f}",
            f"{trade['sell_value']:,.2f}",
            f"{trade['profit_loss']:,.2f}",
            f"{(trade['sell_time'] - trade['buy_time']) / 60:.1f}"
        ])
    print("\n=== Completed Trades ===")
    print(tabulate(completed_trades_table, headers=[
        "Token Name", "Token Address", "Buy Price USD", "Sell Price USD", "Quantity",
        "Capital Spent", "Sell Value", "PnL", "Hold Age (m)"
    ], tablefmt="pretty") if completed_trades_table else "No completed trades.")


def calculate_profit_loss(trades):
    realized_pnl = sum(trade["profit_loss"] for trade in trades["sells"])
    current_holdings_value = sum(trade["current_price"] * trade["quantity"] for trade in trades["buys"])
    capital_spent_on_buys = sum(trade["capital_spent"] for trade in trades["buys"])
    unrealized_pnl = current_holdings_value - capital_spent_on_buys
    return realized_pnl, current_holdings_value, unrealized_pnl


def load_trading_data():
    try:
        with open("trading_data.json", "r") as f:
            data = json.load(f)
            logger.info("Loaded trading_data.json")
            for trade_list in ["buys", "sells", "tracked_sold"]:
                required = REQUIRED_FIELDS[trade_list]
                data[trade_list] = [trade for trade in data.get(trade_list, []) if
                                    all(key in trade for key in required)]
            for trade in data.get("buys", []):
                trade["current_price"] = trade.get("current_price", trade["buy_price"])
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        logger.info("trading_data.json not found or invalid, initializing new data")
        return {"capital": INITIAL_CAPITAL, "buys": [], "sells": [], "tracked_sold": [], "known_tokens": []}


def save_trading_data(trading_data):
    try:
        with open("trading_data.json", "w") as f:
            json.dump(trading_data, f, indent=4)
        logger.info("Saved trading data to trading_data.json")
    except Exception as e:
        logger.error(f"Failed to save trading_data.json: {e}")


def main():
    global FIRST_RUN
    trading_data = load_trading_data()
    current_capital = trading_data["capital"]
    known_tokens = set(trading_data["known_tokens"])
    iteration = 0
    start_time = time.time()

    while current_capital > 0:
        iteration += 1
        runtime_minutes = (time.time() - start_time) / 60
        logger.info(
            f"\nIteration {iteration} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Runtime: {runtime_minutes:.1f} minutes)")

        token_data = get_boosted_tokens()
        if token_data is None:
            time.sleep(60)
            continue
        aggregated_tokens = aggregate_boosts(token_data)
        processed_tokens = process_token_data(aggregated_tokens)
        pair_data_map = {pair["pair_address"]: pair for pair in processed_tokens}

        if not FIRST_RUN:
            # Buy logic
            buy_candidates = sorted(
                [t for t in processed_tokens if t["token_address"] not in known_tokens and
                 t["price_usd"] > 0 and
                 (t["liquidity_usd"] == 0 or t["liquidity_usd"] >= 10000) and
                 t["price_change_5m"] > 5],
                key=lambda x: x["boost_amount"],
                reverse=True
            )[:MAX_BUYS_PER_ITERATION]
            for token in buy_candidates:
                capital_to_spend = current_capital * BUY_PERCENTAGE
                if capital_to_spend > current_capital:
                    break
                quantity = capital_to_spend / token["price_usd"]
                current_capital -= capital_to_spend
                buy_time = time.time()
                trading_data["buys"].append({
                    "token_address": token["token_address"],
                    "pair_address": token["pair_address"],
                    "token_name": token["token_name"],
                    "buy_price": token["price_usd"],
                    "quantity": quantity,
                    "capital_spent": capital_to_spend,
                    "buy_time": buy_time,
                    "highest_price": token["price_usd"],
                    "current_price": token["price_usd"],
                    "trailing_stop_price": token["price_usd"] * (1 - TRAILING_STOP_PERCENT)
                })
                known_tokens.add(token["token_address"])
                logger.info(
                    f"Bought {quantity:,.2f} of {token['token_name']} at ${token['price_usd']:.10f} for ${capital_to_spend:,.2f} (Boost: {token['boost_amount']:.2f}, Change 5m: {token['price_change_5m']:.2f}%)")

        FIRST_RUN = False

        # Sell logic
        for trade in trading_data["buys"][:]:
            pair_address = trade["pair_address"]
            current_price = pair_data_map.get(pair_address, {}).get("price_usd", trade["current_price"])
            trade["current_price"] = current_price
            trade["highest_price"] = max(trade["highest_price"], current_price)
            if current_price > trade["trailing_stop_price"] / (1 - TRAILING_STOP_PERCENT):
                trade["trailing_stop_price"] = current_price * (1 - TRAILING_STOP_PERCENT)
            price_change_5m = pair_data_map.get(pair_address, {}).get("price_change_5m", 0)
            logger.info(
                f"Updated {trade['token_name']}: Current Price ${current_price:.10f}, Change 5m {price_change_5m:.2f}%, Trailing Stop ${trade['trailing_stop_price']:.10f}")

            buy_price = trade["buy_price"]
            hold_age_minutes = (time.time() - trade["buy_time"]) / 60
            sell_reason = None
            sell_fraction = 1.0

            if current_price >= buy_price * SELL_THRESHOLD:
                sell_reason = "Reached 50% profit target"
            elif current_price <= buy_price * STOP_LOSS_THRESHOLD:
                sell_reason = "Hit 30% stop loss"
            elif current_price <= trade["trailing_stop_price"]:
                sell_reason = "Hit 10% trailing stop"
            elif hold_age_minutes > NO_CHANGE_SELL_MINUTES and price_change_5m == 0:
                sell_reason = f"No price change in last {NO_CHANGE_SELL_MINUTES} minutes"
            elif current_price >= buy_price * PARTIAL_SELL_THRESHOLD and trade["quantity"] == trade.get(
                    "original_quantity", trade["quantity"]):
                sell_reason = "Reached 25% profit for partial sell"
                sell_fraction = 0.5
                trade["original_quantity"] = trade["quantity"]

            if sell_reason:
                sell_quantity = trade["quantity"] * sell_fraction
                trade["quantity"] *= (1 - sell_fraction)
                sell_value = current_price * sell_quantity
                capital_spent_fraction = trade["capital_spent"] * sell_fraction
                profit_loss = sell_value - capital_spent_fraction
                current_capital += sell_value
                sell_trade = {
                    "token_address": trade["token_address"],
                    "pair_address": trade["pair_address"],
                    "token_name": trade["token_name"],
                    "buy_price": buy_price,
                    "sell_price": current_price,
                    "quantity": sell_quantity,
                    "capital_spent": capital_spent_fraction,
                    "sell_value": sell_value,
                    "profit_loss": profit_loss,
                    "buy_time": trade["buy_time"],
                    "sell_time": time.time(),
                    "highest_price": trade["highest_price"]
                }
                trading_data["sells"].append(sell_trade)
                if sell_fraction == 1.0:
                    trading_data["tracked_sold"].append({
                        "token_address": trade["token_address"],
                        "pair_address": trade["pair_address"],
                        "token_name": trade["token_name"],
                        "buy_price": buy_price,
                        "sell_price": current_price,
                        "highest_price": trade["highest_price"]
                    })
                    trading_data["buys"].remove(trade)
                else:
                    trade["capital_spent"] *= (1 - sell_fraction)
                logger.info(
                    f"Sold {sell_quantity:,.2f} of {trade['token_name']} at ${current_price:.10f} for ${sell_value:,.2f} (Reason: {sell_reason}, PnL: ${profit_loss:,.2f})")

        display_tokens(processed_tokens, current_capital, trading_data, iteration, runtime_minutes)

        realized_pnl, current_holdings_value, unrealized_pnl = calculate_profit_loss(trading_data)
        print("\n=== Trading Summary ===")
        print(f"Current Capital: ${current_capital:,.2f}")
        print(f"Holdings Value: ${current_holdings_value:,.2f}")
        print(f"Realized P/L: ${realized_pnl:,.2f}")
        print(f"Unrealized P/L: ${unrealized_pnl:,.2f}")
        print(f"Active Trades: {len(trading_data['buys'])}")
        print(f"Completed Trades: {len(trading_data['sells'])}")

        trading_data["capital"] = current_capital
        trading_data["known_tokens"] = list(known_tokens)
        save_trading_data(trading_data)
        time.sleep(60)


if __name__ == "__main__":
    main()