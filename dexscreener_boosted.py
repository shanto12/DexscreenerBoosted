import requests
import json
from datetime import datetime
import time
from tabulate import tabulate
import logging

# Setup logging to a .txt file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# Required fields for each trade list
REQUIRED_FIELDS = {
    "buys": ["token_address", "pair_address", "token_name", "buy_price", "quantity", "capital_spent", "buy_time",
             "highest_price"],
    "sells": ["token_address", "pair_address", "token_name", "buy_price", "sell_price", "quantity", "capital_spent",
              "sell_value", "profit_loss", "buy_time", "sell_time", "highest_price"],
    "tracked_sold": ["token_address", "pair_address", "token_name", "buy_price", "sell_price", "highest_price"]
}


# API Functions
def get_boosted_tokens():
    """Fetch the latest boosted tokens from the DEX Screener API."""
    try:
        response = requests.get(f"{API_BASE_URL}/token-boosts/latest/v1", headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Fetched boosted tokens: {len(data) if isinstance(data, list) else 1} entries")
        return data if isinstance(data, list) else [data]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching boosted tokens: {e}")
        return None


def get_token_pairs(chain_id, token_address):
    """Fetch pair data for a specific token on Solana."""
    try:
        response = requests.get(f"{API_BASE_URL}/token-pairs/v1/{chain_id}/{token_address}", headers=HEADERS,
                                timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching token pairs for {token_address}: {e}")
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
        logger.error(f"Error fetching pair data for {pair_address}: {e}")
        return None


# Helper Functions
def format_price(price):
    """Format a price to show up to 10 decimal places, removing trailing zeros and decimal if no significant digits."""
    if isinstance(price, (int, float)):
        formatted = f"{price:.10f}".rstrip('0').rstrip('.')
        return formatted if formatted else "0"
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
                pair_created_at = pair.get("pairCreatedAt", 0) / 1000
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
                    "buy_vol_24h": "N/A",
                    "sell_vol_24h": "N/A",
                    "price_change_5m": price_change_5m,
                })
    processed_tokens.sort(key=lambda x: x["age_minutes"])
    logger.info(f"Processed {len(processed_tokens)} tokens")
    return processed_tokens


def display_tokens(tokens, current_capital, trades, iteration, runtime_minutes):
    """Display token and trade information in tabulated format."""
    pair_data_map = {pair["pair_address"]: pair for pair in tokens}

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
            token['chain_id']
        ])
    logger.info(f"\nIteration {iteration} Boosted Tokens (Top 10):")
    print(tabulate(boosted_tokens_table, headers=[
        "Age (m)", "Name", "Token Address", "Pair Address", "Boost", "Price USD",
        "Liquidity USD", "FDV", "Market Cap", "Txns (24h)", "Buys (24h)", "Sells (24h)",
        "Buy Vol (24h)", "Sell Vol (24h)", "Price Chg 5m", "Chain"
    ], tablefmt="pretty"))

    logger.info(f"Current Capital: ${current_capital:,.2f}")

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
    logger.info("\nActive Trades:")
    print(tabulate(active_trades_table, headers=[
        "Token Name", "Token Address", "Buy Price USD", "Current Price USD", "Quantity",
        "Capital Spent", "Current Value", "% Change", "PnL", "Highest Price USD",
        "Hold Age (m)", "% Diff to High", "Last Buy (m)"
    ], tablefmt="pretty"))

    completed_trades_table = []
    for trade in trades["sells"]:
        completed_trades_table.append([
            trade['token_name'],
            trade['token_address'],
            format_price(trade["buy_price"]),
            format_price(trade["sell_price"]),
            f"{trade['capital_spent']:,.2f}",
            f"{trade['sell_value']:,.2f}",
            f"{trade['profit_loss']:,.2f}",
            format_price(trade["highest_price"]),
            f"{(trade['sell_time'] - trade['buy_time']) / 60:.1f}"
        ])
    logger.info("\nCompleted Trades:")
    print(tabulate(completed_trades_table, headers=[
        "Token Name", "Token Address", "Buy Price USD", "Sell Price USD", "Capital Spent",
        "Sell Value", "PnL", "Highest Price USD", "Hold Age (m)"
    ], tablefmt="pretty"))

    tracked_sold_table = []
    for trade in trades["tracked_sold"]:
        current_price = pair_data_map.get(trade["pair_address"], {}).get("price_usd", "N/A")
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
    logger.info("\nTracked Sold Tokens:")
    print(tabulate(tracked_sold_table, headers=[
        "Token Name", "Token Address", "Buy Price USD", "Sell Price USD", "Current Price USD",
        "Highest Price USD", "% Apprec to High"
    ], tablefmt="pretty"))


def calculate_profit_loss(trades):
    """Calculate total realized profit/loss and unrealized PnL for active trades."""
    realized_pnl = sum(trade["profit_loss"] for trade in trades["sells"])
    current_holdings_value = sum(trade["current_price"] * trade["quantity"] for trade in trades["buys"])
    capital_spent_on_buys = sum(trade["capital_spent"] for trade in trades["buys"])
    unrealized_pnl = current_holdings_value - capital_spent_on_buys
    return realized_pnl, current_holdings_value, unrealized_pnl


def load_trading_data():
    """Load or initialize trading data from trading_data.json, ensuring all required fields are present."""
    try:
        with open("trading_data.json", "r") as f:
            data = json.load(f)
            logger.info("Loaded trading_data.json")
            for trade_list in ["buys", "sells", "tracked_sold"]:
                required = REQUIRED_FIELDS[trade_list]
                original_trades = data.get(trade_list, [])
                valid_trades = []
                for trade in original_trades:
                    if not isinstance(trade, dict):
                        logger.warning(f"Invalid trade in {trade_list}: {trade} (not a dictionary)")
                        continue
                    if all(key in trade for key in required):
                        valid_trades.append(trade)
                    else:
                        missing = [key for key in required if key not in trade]
                        logger.warning(f"Removed trade from {trade_list} missing fields {missing}: {trade}")
                data[trade_list] = valid_trades
                removed_count = len(original_trades) - len(valid_trades)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} incomplete trades from {trade_list}")
                logger.info(f"Loaded {len(valid_trades)} valid trades into {trade_list}")
            for trade in data.get("buys", []):
                trade["current_price"] = trade.get("current_price", trade["buy_price"])
                trade["price_change_5m"] = trade.get("price_change_5m", 0)
            return data
    except FileNotFoundError:
        logger.info("trading_data.json not found, initializing new data")
        return {
            "capital": INITIAL_CAPITAL,
            "buys": [],
            "sells": [],
            "tracked_sold": [],
            "known_tokens": []
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode trading_data.json: {e}")
        return {
            "capital": INITIAL_CAPITAL,
            "buys": [],
            "sells": [],
            "tracked_sold": [],
            "known_tokens": []
        }


def save_trading_data(trading_data):
    """Save trading data to trading_data.json after validating required fields."""
    for trade_list in ["buys", "sells", "tracked_sold"]:
        required = REQUIRED_FIELDS[trade_list]
        for trade in trading_data.get(trade_list, []):
            if not all(key in trade for key in required):
                missing = [key for key in required if key not in trade]
                logger.error(f"Attempted to save incomplete trade in {trade_list}: missing {missing}")
                raise ValueError(f"Incomplete trade in {trade_list}: missing {missing}")
    try:
        with open("trading_data.json", "w") as f:
            json.dump(trading_data, f, indent=4)
        logger.info("Saved trading data to trading_data.json")
    except Exception as e:
        logger.error(f"Failed to save trading_data.json: {e}")


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
        logger.info(
            f"\nIteration {iteration} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Runtime: {runtime_minutes:.1f} minutes)")

        token_data = get_boosted_tokens()
        if token_data is None:
            time.sleep(60)
            continue
        processed_tokens = process_token_data(token_data)
        pair_data_map = {pair["pair_address"]: pair for pair in processed_tokens}

        if not FIRST_RUN:
            for token in processed_tokens:
                token_address = token["token_address"]
                pair_address = token["pair_address"]
                price_usd = token["price_usd"]
                liquidity_usd = token["liquidity_usd"]
                if token_address not in known_tokens and price_usd > 0 and (
                        liquidity_usd == 0 or liquidity_usd >= 10000):
                    capital_to_spend = current_capital * BUY_PERCENTAGE
                    if capital_to_spend > current_capital:
                        logger.warning(f"Insufficient capital to buy {token['token_name']}")
                        continue
                    quantity = capital_to_spend / price_usd
                    if quantity > 0:
                        current_capital -= capital_to_spend
                        buy_time = time.time()
                        new_trade = {
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
                        }
                        trading_data["buys"].append(new_trade)
                        known_tokens.add(token_address)
                        logger.info(
                            f"Bought {quantity:.2f} of {token['token_name']} at ${price_usd:.10f} for ${capital_to_spend:,.2f}")
        FIRST_RUN = False

        for trade in trading_data["buys"][:]:
            if "pair_address" not in trade:
                logger.warning(f"Skipping trade missing 'pair_address': {trade}")
                trading_data["buys"].remove(trade)
                continue
            pair_address = trade["pair_address"]
            if pair_address in pair_data_map:
                trade["current_price"] = pair_data_map[pair_address]["price_usd"]
                trade["price_change_5m"] = pair_data_map[pair_address]["price_change_5m"]
                trade["highest_price"] = max(trade["highest_price"], trade["current_price"])
            else:
                pass  # Keep last known values

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
                sell_trade = {
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
                }
                trading_data["sells"].append(sell_trade)
                tracked_sold_trade = {
                    "token_address": trade["token_address"],
                    "pair_address": trade["pair_address"],
                    "token_name": trade["token_name"],
                    "buy_price": buy_price,
                    "sell_price": current_price,
                    "highest_price": trade["highest_price"]
                }
                trading_data["tracked_sold"].append(tracked_sold_trade)
                trading_data["buys"].remove(trade)
                logger.info(
                    f"Sold {trade['token_name']} at ${current_price:.10f} for ${sell_value:,.2f} (Reason: {sell_reason})")

        for trade in trading_data["tracked_sold"][:]:
            pair_address = trade["pair_address"]
            if pair_address in pair_data_map:
                current_price = pair_data_map[pair_address]["price_usd"]
                trade["highest_price"] = max(trade["highest_price"], current_price)
                if current_price <= trade["buy_price"] * 0.8:
                    trading_data["tracked_sold"].remove(trade)
                    logger.info(f"Stopped tracking {trade['token_name']} (Price dropped 80%)")

        display_tokens(processed_tokens, current_capital, trading_data, iteration, runtime_minutes)

        realized_pnl, current_holdings_value, unrealized_pnl = calculate_profit_loss(trading_data)
        logger.info(f"\nTrading Summary:")
        logger.info(f"  Current Capital: ${current_capital:,.2f}")
        logger.info(f"  Holdings Value: ${current_holdings_value:,.2f}")
        logger.info(f"  Realized P/L: ${realized_pnl:,.2f}")
        logger.info(f"  Unrealized P/L: ${unrealized_pnl:,.2f}")
        logger.info(f"  Active Trades: {len(trading_data['buys'])}")
        logger.info(f"  Completed Trades: {len(trading_data['sells'])}")

        trading_data["capital"] = current_capital
        trading_data["known_tokens"] = list(known_tokens)
        save_trading_data(trading_data)

        time.sleep(60)

    logger.info("\nCapital depleted. Stopping simulation.")
    realized_pnl, current_holdings_value, unrealized_pnl = calculate_profit_loss(trading_data)
    logger.info(f"Final Trading Summary:")
    logger.info(f"  Final Capital: ${current_capital:,.2f}")
    logger.info(f"  Holdings Value: ${current_holdings_value:,.2f}")
    logger.info(f"  Total Realized P/L: ${realized_pnl:,.2f}")
    logger.info(f"  Total Unrealized P/L: ${unrealized_pnl:,.2f}")
    logger.info(f"  Total Trades: {len(trading_data['sells'])}")


if __name__ == "__main__":
    main()