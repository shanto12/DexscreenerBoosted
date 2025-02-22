import requests
import json
import datetime
import time
from tabulate import tabulate

# Constants
API_BASE_URL = "https://api.dexscreener.com"
BOOSTED_TOKENS_ENDPOINT = "/token-boosts/latest/v1"
TOKEN_PAIRS_ENDPOINT = "/token-pairs/v1"
PAIR_DATA_ENDPOINT = "/latest/dex/pairs"
HEADERS = {
    "User-Agent": "BoostedTokenAnalyzer/1.0",
    "Accept": "application/json"
}
INITIAL_CAPITAL = 10000.0
BUY_PERCENTAGE = 0.10
SELL_THRESHOLD = 1.4
STOP_LOSS_THRESHOLD = 0.7
CHAIN_ID = "solana"

# API Functions
def get_boosted_tokens():
    """Fetch the latest boosted tokens from DEX Screener API."""
    url = f"{API_BASE_URL}{BOOSTED_TOKENS_ENDPOINT}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

def get_token_pairs(chain_id, token_address):
    """Fetch pair data for a specific token on Solana."""
    url = f"{API_BASE_URL}{TOKEN_PAIRS_ENDPOINT}/{chain_id}/{token_address}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

def get_pair_data(chain_id, pair_addresses):
    """Fetch detailed pair data for multiple pair addresses."""
    url = f"{API_BASE_URL}{PAIR_DATA_ENDPOINT}/{chain_id}/{','.join(pair_addresses)}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

# Helper Functions
def format_price(price):
    """Format a price to 10 decimal places, stripping trailing zeros."""
    if price == "N/A" or not isinstance(price, (int, float)):
        return "N/A"
    return f"{price:.10f}".rstrip('0').rstrip('.')

def format_currency(value):
    """Format monetary values with commas."""
    if value == "N/A" or not isinstance(value, (int, float)):
        return "N/A"
    return f"${value:,.2f}"

def process_token_data(boosted_tokens):
    """Process and deduplicate token data, filtering for Solana tokens."""
    current_time = int(time.time())
    token_info = {}
    for token in boosted_tokens:
        if token.get('chainId') == CHAIN_ID:
            token_address = token['tokenAddress']
            if token_address not in token_info or token['amount'] > token_info[token_address]['boost_amount']:
                token_info[token_address] = {
                    'boost_amount': token['amount'],
                    'description': token.get('description', '')
                }

    all_pairs = []
    for token_address, info in token_info.items():
        pairs = get_token_pairs(CHAIN_ID, token_address)
        if pairs:
            all_pairs.extend([(token_address, pair['pairAddress'], info) for pair in pairs])

    batch_size = 30
    detailed_pairs = []
    for i in range(0, len(all_pairs), batch_size):
        batch = all_pairs[i:i + batch_size]
        pair_addresses = [pair[1] for pair in batch]
        pair_data = get_pair_data(CHAIN_ID, pair_addresses)
        if pair_data and 'pairs' in pair_data:
            detailed_pairs.extend(pair_data['pairs'])

    processed_tokens = []
    for pair in detailed_pairs:
        token_address = next((t[0] for t in all_pairs if t[1] == pair['pairAddress']), None)
        if not token_address or token_address not in token_info:
            continue
        info = token_info[token_address]
        pair_created_at = pair.get('pairCreatedAt', current_time * 1000) / 1000
        age_minutes = (current_time - pair_created_at) / 60
        description = info['description']
        token_name = description.split()[0] if description else "Unknown"
        price_usd = float(pair.get('priceUsd', "N/A")) if pair.get('priceUsd', "N/A") != "N/A" else "N/A"
        liquidity_usd = pair.get('liquidity', {}).get('usd', "N/A")
        processed_tokens.append({
            'age_minutes': age_minutes,
            'token_address': token_address,
            'chain_id': CHAIN_ID,
            'boost_amount': info['boost_amount'],
            'pair_address': pair['pairAddress'],
            'token_name': token_name,
            'price_usd': price_usd,
            'liquidity_usd': liquidity_usd,
            'fdv': pair.get('fdv', "N/A"),
            'market_cap': pair.get('marketCap', "N/A"),
            'txns_24h': pair.get('txns', {}).get('h24', {}).get('buys', 0) + pair.get('txns', {}).get('h24', {}).get('sells', 0),
            'buys_24h': pair.get('txns', {}).get('h24', {}).get('buys', 0),
            'sells_24h': pair.get('txns', {}).get('h24', {}).get('sells', 0),
            'volume_24h': pair.get('volume', {}).get('h24', "N/A"),
            'price_change_5m': pair.get('priceChange', {}).get('m5', "N/A"),
            'price_change_1h': pair.get('priceChange', {}).get('h1', "N/A"),
            'price_change_6h': pair.get('priceChange', {}).get('h6', "N/A"),
            'price_change_24h': pair.get('priceChange', {}).get('h24', "N/A")
        })

    processed_tokens.sort(key=lambda x: x['age_minutes'])
    return processed_tokens

def display_tokens(tokens, current_capital, trades, iteration, runtime_minutes):
    """Display token and trade information in tabulated format."""
    current_time = int(time.time())
    print(f"\nIteration {iteration} - Runtime: {runtime_minutes:.2f} minutes")
    print(f"Current Capital: {format_currency(current_capital)}")

    # Boosted Tokens
    token_headers = ["Age (m)", "Name", "Token Address", "Pair Address", "Boost", "Price USD", "Liquidity USD",
                     "FDV", "Market Cap", "Txns (24h)", "Buys (24h)", "Sells (24h)", "Buy Vol (24h)", "Sell Vol (24h)",
                     "Price Chg 5m", "1h", "6h", "24h", "Chain"]
    token_rows = [[f"{t['age_minutes']:.1f}", t['token_name'], t['token_address'], t['pair_address'], t['boost_amount'],
                   format_price(t['price_usd']), format_currency(t['liquidity_usd']), format_currency(t['fdv']),
                   format_currency(t['market_cap']), t['txns_24h'], t['buys_24h'], t['sells_24h'],
                   format_currency(t['volume_24h']), "N/A", t['price_change_5m'], t['price_change_1h'],
                   t['price_change_6h'], t['price_change_24h'], t['chain_id']] for t in tokens[:10]]
    print("\nBoosted Tokens:")
    print(tabulate(token_rows, headers=token_headers, tablefmt="pretty",
                   maxcolwidths=[10, 20, 45, 45, 10, 12, 15, 15, 15, 10, 10, 10, 15, 15, 10, 10, 10, 10, 10]))

    # Active Trades
    active_headers = ["Token Name", "Token Address", "Buy Price USD", "Current Price USD", "Quantity", "Capital Spent",
                      "Current Value", "% Change", "PnL", "Highest Price USD", "Hold Age (m)", "% Diff to High", "Last Buy (m)"]
    active_rows = []
    for trade in trades['buys']:
        current_price = next((t['price_usd'] for t in tokens if t['pair_address'] == trade['pair_address']), trade['highest_price'])
        hold_age = (current_time - trade['buy_time']) / 60
        pc_5m = next((t['price_change_5m'] for t in tokens if t['pair_address'] == trade['pair_address']), 0)
        last_buy = 5.0 if pc_5m == 0 else min(5.0, 5.0 / (1 + abs(pc_5m / 100)))
        current_value = current_price * trade['quantity'] if current_price != "N/A" else "N/A"
        percent_change = ((current_price - trade['buy_price']) / trade['buy_price'] * 100) if current_price != "N/A" else "N/A"
        pnl = current_value - trade['capital_spent'] if current_value != "N/A" else "N/A"
        diff_to_high = ((trade['highest_price'] - trade['buy_price']) / trade['buy_price'] * 100) if trade['buy_price'] > 0 else "N/A"
        active_rows.append([trade['token_name'], trade['token_address'], format_price(trade['buy_price']),
                            format_price(current_price), f"{trade['quantity']:.2f}", format_currency(trade['capital_spent']),
                            format_currency(current_value), f"{percent_change:.2f}" if percent_change != "N/A" else "N/A",
                            format_currency(pnl), format_price(trade['highest_price']), f"{hold_age:.1f}",
                            f"{diff_to_high:.2f}" if diff_to_high != "N/A" else "N/A", f"{last_buy:.1f}"])
    if active_rows:
        print("\nActive Trades:")
        print(tabulate(active_rows, headers=active_headers, tablefmt="pretty",
                       maxcolwidths=[20, 45, 15, 15, 10, 15, 15, 10, 15, 15, 10, 15, 10]))

    # Completed Trades
    completed_headers = ["Token Name", "Token Address", "Buy Price USD", "Sell Price USD", "Capital Spent", "Sell Value",
                         "PnL", "Highest Price USD", "Hold Age (m)"]
    completed_rows = [[t['token_name'], t['token_address'], format_price(t['buy_price']), format_price(t['sell_price']),
                       format_currency(t['capital_spent']), format_currency(t['sell_value']), format_currency(t['profit_loss']),
                       format_price(t['highest_price']), f"{(t['sell_time'] - t['buy_time']) / 60:.1f}"]
                      for t in trades['sells']]
    if completed_rows:
        print("\nCompleted Trades:")
        print(tabulate(completed_rows, headers=completed_headers, tablefmt="pretty",
                       maxcolwidths=[20, 45, 15, 15, 15, 15, 15, 15, 15]))

    # Tracked Sold Tokens
    tracked_headers = ["Token Name", "Token Address", "Buy Price USD", "Sell Price USD", "Current Price USD",
                       "Highest Price USD", "% Apprec to High"]
    tracked_rows = []
    for t in trades['tracked_sold']:
        current_price = next((tok['price_usd'] for tok in tokens if tok['pair_address'] == t['pair_address']), t['highest_price'])
        apprec_to_high = ((t['highest_price'] - t['buy_price']) / t['buy_price'] * 100) if t['buy_price'] > 0 else "N/A"
        tracked_rows.append([t['token_name'], t['token_address'], format_price(t['buy_price']),
                             format_price(t['sell_price']), format_price(current_price), format_price(t['highest_price']),
                             f"{apprec_to_high:.2f}" if apprec_to_high != "N/A" else "N/A"])
    if tracked_rows:
        print("\nTracked Sold Tokens:")
        print(tabulate(tracked_rows, headers=tracked_headers, tablefmt="pretty",
                       maxcolwidths=[20, 45, 15, 15, 15, 15, 15]))

def calculate_profit_loss(trades):
    """Calculate total realized and unrealized profit/loss."""
    realized_pnl = sum(t['profit_loss'] for t in trades['sells'])
    current_value = sum(t['quantity'] * next((tok['price_usd'] for tok in processed_tokens if tok['pair_address'] == t['pair_address']), 0)
                        for t in trades['buys'] if isinstance(next((tok['price_usd'] for tok in processed_tokens if tok['pair_address'] == t['pair_address']), 0), (int, float)))
    capital_spent = sum(t['capital_spent'] for t in trades['buys'])
    unrealized_pnl = current_value - capital_spent if current_value > 0 else 0
    return realized_pnl, current_value, unrealized_pnl

def calculate_avg_percent_increase(trades, processed_tokens):
    """Calculate average percentage increase to highest price."""
    all_trades = trades['buys'] + trades['sells'] + trades['tracked_sold']
    increases = []
    to_remove = []
    for trade in all_trades:
        if trade['buy_price'] <= 0:
            continue
        current_price = next((t['price_usd'] for t in processed_tokens if t['pair_address'] == trade['pair_address']), trade['highest_price'])
        if current_price != "N/A" and current_price <= trade['buy_price'] * 0.2:
            to_remove.append(trade['token_address'])
            continue
        percent_increase = ((trade['highest_price'] - trade['buy_price']) / trade['buy_price']) * 100
        increases.append(percent_increase)
    avg_increase = sum(increases) / len(increases) if increases else 0
    return avg_increase, to_remove

def load_trading_data():
    """Load or initialize trading data from JSON."""
    try:
        with open('trading_data.json', 'r') as f:
            data = json.load(f)
            data.setdefault('capital', INITIAL_CAPITAL)
            data.setdefault('buys', [])
            data.setdefault('sells', [])
            data.setdefault('tracked_sold', [])
            data.setdefault('known_tokens', [])
            for trade in data['buys'] + data['sells'] + data['tracked_sold']:
                trade.setdefault('highest_price', trade['buy_price'])
                trade.setdefault('buy_time', int(time.time()))
                if 'sell_time' in trade:
                    trade.setdefault('quantity', 1.0)
            return data
    except FileNotFoundError:
        return {'capital': INITIAL_CAPITAL, 'buys': [], 'sells': [], 'tracked_sold': [], 'known_tokens': []}

def save_trading_data(trading_data):
    """Save trading data to JSON."""
    with open('trading_data.json', 'w') as f:
        json.dump(trading_data, f, indent=4)

# Main Logic
def main():
    """Main function to monitor and simulate trading."""
    FIRST_RUN = True
    trading_data = load_trading_data()
    current_capital = trading_data['capital']
    known_tokens = set(trading_data['known_tokens'])
    iteration = 0
    start_time = time.time()
    global processed_tokens  # To use in calculate_profit_loss

    while current_capital > 0:
        iteration += 1
        runtime_minutes = (time.time() - start_time) / 60
        print(f"\nTimestamp: {datetime.datetime.now()}, Iteration: {iteration}, Runtime: {runtime_minutes:.2f} minutes")

        boosted_tokens = get_boosted_tokens()
        if not boosted_tokens:
            print("Failed to fetch boosted tokens. Retrying in 60 seconds.")
            time.sleep(60)
            continue

        processed_tokens = process_token_data(boosted_tokens)

        if not FIRST_RUN:
            for token in processed_tokens:
                if (token['token_address'] not in known_tokens and token['price_usd'] != "N/A" and token['price_usd'] > 0 and
                    (token['liquidity_usd'] == "N/A" or token['liquidity_usd'] == 0 or token['liquidity_usd'] >= 10000)):
                    capital_to_spend = current_capital * BUY_PERCENTAGE
                    if capital_to_spend > current_capital:
                        print(f"Insufficient capital to buy {token['token_name']} ({token['token_address']})")
                        continue
                    quantity = capital_to_spend / token['price_usd']
                    if quantity <= 0:
                        print(f"Invalid quantity for {token['token_name']} ({token['token_address']})")
                        continue
                    trade = {
                        'token_name': token['token_name'],
                        'token_address': token['token_address'],
                        'pair_address': token['pair_address'],
                        'buy_price': token['price_usd'],
                        'highest_price': token['price_usd'],
                        'quantity': quantity,
                        'capital_spent': capital_to_spend,
                        'buy_time': int(time.time())
                    }
                    trading_data['buys'].append(trade)
                    current_capital -= capital_to_spend
                    known_tokens.add(token['token_address'])
                    print(f"Bought {quantity:.2f} of {token['token_name']} ({token['token_address']}) at {format_price(token['price_usd'])} for {format_currency(capital_to_spend)}")
        else:
            FIRST_RUN = False

        sells_to_remove = []
        for i, trade in enumerate(trading_data['buys']):
            current_price = next((t['price_usd'] for t in processed_tokens if t['pair_address'] == trade['pair_address']), trade['highest_price'])
            if current_price != "N/A":
                trade['highest_price'] = max(trade['highest_price'], current_price)
                pc_5m = next((t['price_change_5m'] for t in processed_tokens if t['pair_address'] == trade['pair_address']), "N/A")
                if current_price >= trade['buy_price'] * SELL_THRESHOLD:
                    reason = "Reached profit target"
                elif current_price <= trade['buy_price'] * STOP_LOSS_THRESHOLD:
                    reason = "Hit stop loss"
                elif pc_5m == 0:
                    reason = "No buy activity in last 5 minutes"
                else:
                    continue
                sell_value = current_price * trade['quantity']
                profit_loss = sell_value - trade['capital_spent']
                trade.update({
                    'sell_price': current_price,
                    'sell_value': sell_value,
                    'profit_loss': profit_loss,
                    'sell_time': int(time.time())
                })
                current_capital += sell_value
                trading_data['sells'].append(trade)
                trading_data['tracked_sold'].append(trade.copy())
                sells_to_remove.append(i)
                print(f"Sold {trade['quantity']:.2f} of {trade['token_name']} ({trade['token_address']}) at {format_price(current_price)} for {format_currency(sell_value)}, PnL: {format_currency(profit_loss)}, Reason: {reason}")

        for idx in sorted(sells_to_remove, reverse=True):
            trading_data['buys'].pop(idx)

        tracked_to_remove = []
        for i, trade in enumerate(trading_data['tracked_sold']):
            current_price = next((t['price_usd'] for t in processed_tokens if t['pair_address'] == trade['pair_address']), trade['highest_price'])
            if current_price != "N/A":
                trade['highest_price'] = max(trade['highest_price'], current_price)
                if current_price <= trade['buy_price'] * 0.2:
                    tracked_to_remove.append(i)
                    print(f"Removed {trade['token_name']} ({trade['token_address']}) from tracked_sold: Price dropped 80%")

        for idx in sorted(tracked_to_remove, reverse=True):
            trading_data['tracked_sold'].pop(idx)

        avg_increase, tokens_to_remove = calculate_avg_percent_increase(trading_data, processed_tokens)
        known_tokens.difference_update(tokens_to_remove)

        display_tokens(processed_tokens, current_capital, trading_data, iteration, runtime_minutes)

        realized_pnl, holdings_value, unrealized_pnl = calculate_profit_loss(trading_data)
        print(f"\nTrading Summary:")
        print(f"Current Capital: {format_currency(current_capital)}")
        print(f"Current Holdings Value: {format_currency(holdings_value)}")
        print(f"Realized P/L: {format_currency(realized_pnl)}")
        print(f"Unrealized P/L: {format_currency(unrealized_pnl)}")
        print(f"Active Trades: {len(trading_data['buys'])}")
        print(f"Completed Trades: {len(trading_data['sells'])}")
        print(f"Average % Increase to High: {avg_increase:.2f}")

        save_trading_data(trading_data)
        time.sleep(60)

    print("\nCapital depleted. Final Report:")
    realized_pnl, holdings_value, unrealized_pnl = calculate_profit_loss(trading_data)
    print(f"Final Capital: {format_currency(current_capital)}")
    print(f"Final Holdings Value: {format_currency(holdings_value)}")
    print(f"Total Realized P/L: {format_currency(realized_pnl)}")
    print(f"Total Unrealized P/L: {format_currency(unrealized_pnl)}")
    print(f"Total Active Trades: {len(trading_data['buys'])}")
    print(f"Total Completed Trades: {len(trading_data['sells'])}")
    avg_increase, _ = calculate_avg_percent_increase(trading_data, processed_tokens)
    print(f"Final Average % Increase to High: {avg_increase:.2f}")

if __name__ == "__main__":
    main()