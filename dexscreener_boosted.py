import requests
import json
from datetime import datetime
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


def get_boosted_tokens():
    """Fetch the latest boosted tokens from DEX Screener API"""
    try:
        response = requests.get(BOOSTS_ENDPOINT, headers=HEADERS)
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
        response = requests.get(endpoint, headers=HEADERS)
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
        response = requests.get(endpoint, headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching pair data: {str(e)}")
        return None


def process_token_data(token_data):
    """Process token data and extract relevant information"""
    processed_tokens = []
    current_time = int(time.time())  # Current timestamp in seconds

    for token in token_data:
        # Try to find a timestamp in the boosted token data (if available)
        # If not, use pairCreatedAt from pair data as a fallback
        boost_time = token.get("boostTimestamp", None)  # Try to find a timestamp in boosted data
        pair_address = token.get("pairAddress", None)  # Try to get pair address from boosted data

        # If pairAddress is not in boosted data, fetch it using token-pairs endpoint
        if not pair_address:
            token_pairs = get_token_pairs(token.get("chainId", "solana"), token.get("tokenAddress", ""))
            if token_pairs and len(token_pairs) > 0:
                pair_address = token_pairs[0].get("pairAddress", "N/A")

        # Use pair_address to fetch detailed pair data
        pair_data = get_pair_data(token.get("chainId", "solana"),
                                  pair_address) if pair_address and pair_address != "N/A" else None

        if pair_data and pair_data.get("pairs") and len(pair_data["pairs"]) > 0:
            primary_pair = pair_data["pairs"][0]
            boost_time = primary_pair.get("pairCreatedAt", current_time) / 1000  # Convert milliseconds to seconds

        # Calculate age in minutes, ensuring it's positive
        age_seconds = max(0, current_time - (boost_time if isinstance(boost_time, (int, float)) else current_time))
        age_minutes = age_seconds / 60  # Convert to minutes

        token_info = {
            "age_minutes": age_minutes,  # Changed to minutes
            "token_address": token.get("tokenAddress", "N/A"),
            "chain_id": token.get("chainId", "N/A"),
            "boost_amount": token.get("amount", 0),
            "total_boost_amount": token.get("totalAmount", 0),
            "pair_address": pair_address if pair_address and pair_address != "N/A" else "N/A",
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

        # Populate pair data if available
        if pair_data and pair_data.get("pairs") and len(pair_data["pairs"]) > 0:
            primary_pair = pair_data["pairs"][0]
            token_info["pair_address"] = primary_pair.get("pairAddress", "N/A")
            token_info["token_name"] = primary_pair.get("baseToken", {}).get("name", token_info["token_name"])

            # Handle priceUsd (convert string to float if possible)
            price_usd = primary_pair.get("priceUsd", "N/A")
            if price_usd != "N/A" and isinstance(price_usd, str):
                try:
                    token_info["price_usd"] = float(price_usd)
                except ValueError:
                    token_info["price_usd"] = "N/A"
                    print(
                        f"Warning: Invalid priceUsd format for token {token_info['token_name']} - {token_info['token_address']}")
                    print("Pair Data:", json.dumps(primary_pair, indent=2))
            elif isinstance(price_usd, (int, float)):
                token_info["price_usd"] = price_usd
            else:
                token_info["price_usd"] = "N/A"
                print(
                    f"Warning: priceUsd not found or invalid for token {token_info['token_name']} - {token_info['token_address']}")
                print("Pair Data:", json.dumps(primary_pair, indent=2))

            token_info["liquidity_usd"] = primary_pair.get("liquidity", {}).get("usd", 0)
            token_info["fdv"] = primary_pair.get("fdv", 0)
            token_info["market_cap"] = primary_pair.get("marketCap", 0)
            token_info["volume_24h"] = primary_pair.get("volume", {}).get("h24", 0)

            # Handle txns data (use h24 for total, or sum buys/sells)
            txns_data = primary_pair.get("txns", {})
            token_info["txns_24h"] = txns_data.get("h24", {}).get("buys", 0) + txns_data.get("h24", {}).get("sells",
                                                                                                            0)  # Sum buys and sells for 24h
            token_info["buys_24h"] = txns_data.get("h24", {}).get("buys", 0)  # Number of buys in 24h
            token_info["sells_24h"] = txns_data.get("h24", {}).get("sells", 0)  # Number of sells in 24h

            # Handle volume data (use h24 for buy and sell volumes if available)
            volume_data = primary_pair.get("volume", {})
            # If buyH24 and sellH24 aren't available, we might need to estimate or leave as 0
            token_info["buy_volume_24h"] = volume_data.get("buyH24", 0)  # Buy volume in 24h (if available)
            token_info["sell_volume_24h"] = volume_data.get("sellH24", 0)  # Sell volume in 24h (if available)
            if token_info["buy_volume_24h"] == 0 or token_info["sell_volume_24h"] == 0:
                # Fallback: Distribute total h24 volume proportionally based on buy/sell counts
                total_volume_h24 = volume_data.get("h24", 0)
                total_txns_h24 = token_info["buys_24h"] + token_info["sells_24h"]
                if total_txns_h24 > 0 and total_volume_h24 > 0:
                    token_info["buy_volume_24h"] = (token_info["buys_24h"] / total_txns_h24) * total_volume_h24
                    token_info["sell_volume_24h"] = (token_info["sells_24h"] / total_txns_h24) * total_volume_h24

            # Price changes
            price_change_data = primary_pair.get("priceChange", {})
            token_info["price_change_5m"] = price_change_data.get("m5", 0)
            token_info["price_change_1h"] = price_change_data.get("h1", 0)
            token_info["price_change_6h"] = price_change_data.get("h6", 0)
            token_info["price_change_24h"] = price_change_data.get("h24", 0)

        # Debug age calculation
        if token_info["age_minutes"] == 0:
            print(f"Warning: Age is 0 for token {token_info['token_name']} - {token_info['token_address']}")
            print(f"Boost Time: {boost_time}, Current Time: {current_time}")

        processed_tokens.append(token_info)

    return processed_tokens


def display_tokens(tokens):
    """Display token information in a formatted table using tabulate, limited to first 10 coins"""
    table_data = []
    for token in tokens[:10]:  # Limit to first 10 coins
        row = [
            f"{token['age_minutes']:.2f}m",  # Display age in minutes with 2 decimal places
            token.get("token_name", "Unknown")[:20],
            token['token_address'][:45],
            token.get("pair_address", "N/A")[:45],
            f"{token['boost_amount']:,}",
            f"${float(token['price_usd']):.5f}" if token.get("price_usd") != "N/A" and isinstance(token["price_usd"], (
            int, float)) else "N/A",
            f"${token['liquidity_usd']:,.0f}",
            f"${token['fdv']:,.0f}",
            f"${token['market_cap']:,.0f}",
            token['txns_24h'],  # Use total txns for 24h
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

    # Sort by age (newest first, i.e., lowest minutes)
    table_data.sort(key=lambda x: float(x[0].replace("m", "")))

    headers = ["Age (m)", "Name", "Token Address", "Pair Address", "Boost", "Price USD", "Liquidity USD", "FDV",
               "Market Cap",
               "Txns (24h)", "Buys (24h)", "Sells (24h)", "Buy Vol (24h)", "Sell Vol (24h)",
               "Price Chg 5m", "Price Chg 1h", "Price Chg 6h", "Price Chg 24h", "Chain"]
    print("\n=== Boosted Tokens (Sorted by Age - Newest First) ===")
    print(tabulate(table_data, headers=headers, tablefmt="pretty",
                   maxcolwidths=[10, 20, 45, 45, 10, 12, 15, 15, 15, 10, 10, 10, 15, 15, 10, 10, 10, 10, 10]))


def main():
    # Get boosted tokens
    boosted_tokens = get_boosted_tokens()
    if not boosted_tokens:
        print("Failed to fetch boosted tokens.")
        return

    # Process the data
    processed_tokens = process_token_data(boosted_tokens)

    # Sort by age (newest first)
    sorted_tokens = sorted(processed_tokens, key=lambda x: x["age_minutes"])

    # Display results
    display_tokens(sorted_tokens)

    # Save to JSON file for later analysis
    with open("boosted_tokens.json", "w") as f:
        json.dump(sorted_tokens, f, indent=2)
    print("Data saved to 'boosted_tokens.json'")


if __name__ == "__main__":
    main()