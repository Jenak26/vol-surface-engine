# data/fetcher.py
import pandas as pd
import numpy as np

try:
    from nsepython import nse_optionchain_scrapper
    NSE_AVAILABLE = True
except ImportError:
    NSE_AVAILABLE = False


def fetch_nse_option_chain(symbol: str = 'NIFTY') -> pd.DataFrame:
    """
    Fetch live NSE options chain for a given symbol.

    Returns DataFrame with columns:
        strike, expiry, option_type, last_price, bid, ask, iv, oi, volume

    Falls back to synthetic data if nsepython is unavailable or NSE is unreachable.
    """
    if NSE_AVAILABLE:
        try:
            raw = nse_optionchain_scrapper(symbol)
            df = _parse_nse_raw(raw)
            if len(df) > 0 and 'last_price' in df.columns:
                return df
        except Exception:
            pass  # fall through to synthetic data

    return _generate_synthetic_chain()


def _parse_nse_raw(raw: dict) -> pd.DataFrame:
    records = []
    for entry in raw.get('records', {}).get('data', []):
        expiry = entry.get('expiryDate')
        strike = entry.get('strikePrice')
        for otype in ('CE', 'PE'):
            data = entry.get(otype, {})
            if not data:
                continue
            records.append({
                'strike':      float(strike),
                'expiry':      expiry,
                'option_type': 'call' if otype == 'CE' else 'put',
                'last_price':  float(data.get('lastPrice', 0)),
                'bid':         float(data.get('bidprice', 0)),
                'ask':         float(data.get('askPrice', 0)),
                'iv':          float(data.get('impliedVolatility', 0)) / 100,
                'oi':          int(data.get('openInterest', 0)),
                'volume':      int(data.get('totalTradedVolume', 0)),
            })
    return pd.DataFrame(records)


def _generate_synthetic_chain() -> pd.DataFrame:
    """
    Generate a synthetic options chain that mimics a realistic volatility smile.
    Used for development and testing when NSE data is unavailable.

    Smile shape: ATM vol = 0.20, with wings (OTM puts) elevated by skew.
    """
    spot = 22000.0
    strikes = np.arange(19000, 25500, 500, dtype=float)
    expiries = ['25-Jun-2026', '31-Jul-2026', '28-Aug-2026']
    T_values = [30/365, 67/365, 95/365]

    r = 0.065  # India repo rate ≈ 6.5%
    atm_vol = 0.20
    skew = -0.15   # negative skew: lower strikes have higher vol
    smile = 0.05   # wing curvature

    records = []
    for expiry, T in zip(expiries, T_values):
        for strike in strikes:
            log_moneyness = np.log(strike / spot)
            sigma = atm_vol + skew * log_moneyness + smile * log_moneyness ** 2
            sigma = max(sigma, 0.08)

            from src.black_scholes import bs_price as _bs
            call_price = _bs(spot, strike, T, r, sigma, 'call')
            put_price  = _bs(spot, strike, T, r, sigma, 'put')

            for otype, price in [('call', call_price), ('put', put_price)]:
                spread = price * 0.02  # 2% bid-ask
                records.append({
                    'strike':      strike,
                    'expiry':      expiry,
                    'option_type': otype,
                    'last_price':  price,
                    'bid':         price - spread / 2,
                    'ask':         price + spread / 2,
                    'iv':          sigma,
                    'oi':          int(np.random.default_rng(int(strike)).integers(100, 10000)),
                    'volume':      int(np.random.default_rng(int(strike + 1)).integers(10, 1000)),
                    'spot':        spot,
                    'T':           T,
                    'r':           r,
                })
    return pd.DataFrame(records)
