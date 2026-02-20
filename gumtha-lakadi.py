import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import numpy as np
from datetime import datetime
import pytz
import hmac

# ==============================
# PASSWORD LOGIN (TOP OF FILE)
# ==============================

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["PASSWORD"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" in st.session_state:
        return st.session_state["password_correct"]

    st.text_input("Enter password to access NSE scalper", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()

# ==============================
# STOCK LIST (Nifty‑50 + extras)
# ==============================

nifty_plus_extra = [
    # Nifty 50 subset (you can extend to full 50)
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','HINDUNILVR.NS','ICICIBANK.NS',
    'KOTAKBANK.NS','BHARTIARTL.NS','ITC.NS','LT.NS','AXISBANK.NS','ASIANPAINT.NS',
    'SUNPHARMA.NS','TITAN.NS','NESTLEIND.NS','HCLTECH.NS','TECHM.NS',
    'HINDALCO.NS','ULTRACEMCO.NS','POWERGRID.NS','NTPC.NS','CIPLA.NS','ONGC.NS',
    'TATAMOTORS.NS','COALINDIA.NS','JSWSTEEL.NS','WIPRO.NS','BPCL.NS','DRREDDY.NS',
    'DIVISLAB.NS','APOLLOHOSP.NS','BAJFINANCE.NS','EICHERMOT.NS','ADANIENT.NS',
    # Extra stocks
    'ETERNAL.NS',      # Eternal (Zomato)
    'HDFCLIFE.NS',     # HDFC Life
    'MARUTI.NS',       # Maruti Suzuki
    'BOSCHLTD.NS',     # Bosch
]

# ==============================
# DATA FETCH & INDICATORS
# ==============================

@st.cache_data(ttl=15)
def get_data(symbol, interval='1m'):
    try:
        data = yf.download(symbol, period='1d', interval=interval, progress=False)
        if len(data) < 20: return None

        data['EMA9']  = EMAIndicator(data['Close'], 9).ema_indicator()
        data['EMA21'] = EMAIndicator(data['Close'], 21).ema_indicator()
        data['EMA50'] = EMAIndicator(data['Close'], 50).ema_indicator()
        data['RSI']   = RSIIndicator(data['Close']).rsi()
        data['VWAP']  = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        data['Vol_MA'] = data['Volume'].rolling(20).mean()
        data['ATR']   = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()

        macd = MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        return data
    except:
        return None

def is_high_probability_time():
#    ist = pytz.timezone('Asia/Kolkata')
#    now = datetime.now(ist)
#    hour, minute = now.hour, now.minute
#    time_min = hour * 60 + minute
#    return (570 <= time_min <= 660) or (840 <= time_min <= 900)  # 9:30–11:00, 14:00–15:00
return True  # For testing, disable time check

def evaluate_conditions(latest):
    """6 conditions for probability tiers."""
    return [
        latest['EMA9'] > latest['EMA21'] > latest['EMA50'],   # 1. Triple EMA stack
        55 <= latest['RSI'] <= 75,                            # 2. RSI gold zone
        latest['Close'] > latest['VWAP'],                     # 3. Above VWAP
        latest['Volume'] > latest['Vol_MA'] * 1.5,            # 4. Volume surge
        latest['MACD'] > latest['MACD_signal'],               # 5. MACD bullish
        latest['ATR'] / latest['Close'] < 0.015               # 6. Low noise ATR <1.5%
    ]

def classify_probability(score, is_bullish):
    if score == 6:
        return "🟢 80% LONG" if is_bullish else "🔴 80% SHORT", 0.8, 1.67
    elif score == 5:
        return "🟡 70% LONG" if is_bullish else "🟠 70% SHORT", 1.0, 1.50
    elif score == 4:
        return "🔵 60% LONG" if is_bullish else "🟣 60% SHORT", 1.2, 1.30
    else:
        return "⚪ NO TRADE", None, None

def get_multiframe_signal(symbol):
    """Check 5‑min vs 15‑min trend for Multi‑TF column."""
    data_5m = get_data(symbol, '5m')
    data_15m = get_data(symbol, '15m')
    if data_5m is None or data_15m is None: return "N/A"

    latest_5m = data_5m.iloc[-1]
    latest_15m = data_15m.iloc[-1]

    tf_5m = "LONG" if latest_5m['EMA9'] > latest_5m['EMA21'] else "SHORT"
    tf_15m = "LONG" if latest_15m['EMA9'] > latest_15m['EMA21'] else "SHORT"

    return f"5m:{tf_5m} | 15m:{tf_15m}"

def calculate_position(entry, atr, capital, risk_pct, rr=1.5, side="LONG"):
    """Shares, SL, TP for both LONG and SHORT."""
    stop_dist = atr * 0.6
    tp_dist   = stop_dist * rr

    risk_amount = capital * (risk_pct / 100)
    shares = int(risk_amount / stop_dist) if stop_dist > 0 else 0
    shares = max(1, shares)

    if side == "LONG":
        sl = entry - stop_dist
        tp = entry + tp_dist
    else:  # SHORT
        sl = entry + stop_dist
        tp = entry - tp_dist

    return shares, sl, tp, rr

# ==============================
# MAIN APP
# ==============================

st.set_page_config(page_title="NSE 60/70/80% + Eternal/HDFCLIFE/MARUTI/BOSCH", layout="wide")
st.title("🎯 NSE Intraday: Nifty‑50 + Eternal, HDFC Life, Maruti, Bosch")

if not is_high_probability_time():
    st.error("⏰ Outside high‑probability hours (9:30–11:00 or 14:00–15:00 IST)")
    st.stop()

st.sidebar.header("Risk & Tiers")
capital = st.sidebar.number_input("Capital (₹)", 100000, step=5000)
auto_refresh = st.sidebar.checkbox("Auto‑refresh 15s", True)

progress = st.progress(0)
prices_1m = {}
trades = []

for i, sym in enumerate(nifty_plus_extra):
    data_1m = get_data(sym, '1m')
    prices_1m[sym] = data_1m

    if data_1m is not None:
        latest = data_1m.iloc[-1]

        # Determine trend direction
        is_bullish = latest['EMA9'] > latest['EMA21']

        # Evaluate 6‑condition score
        conditions = evaluate_conditions(latest)
        score = sum(conditions)

        signal, risk_pct, rr = classify_probability(score, is_bullish)
        if signal == "⚪ NO TRADE":
            continue

        side = "LONG" if "LONG" in signal else "SHORT"
        shares, sl, tp, _ = calculate_position(latest['Close'], latest['ATR'], capital, risk_pct, rr, side)

        # Multi‑timeframe column
        multiframe = get_multiframe_signal(sym)

        trades.append({
            'Stock': sym.replace('.NS',''),
            'Price': latest['Close'],
            'Side': side,
            'Signal': signal,
            'Shares': shares,
            'SL': sl,
            'TP': tp,
            'Risk ₹': shares * abs(latest['Close'] - sl),
            'Profit ₹': shares * abs(tp - latest['Close']),
            'RSI': latest['RSI'],
            'Score': score,
            'Multi‑TF': multiframe
        })
    progress.progress((i+1)/len(nifty_plus_extra))

# ==============================
# DISPLAY TABLE & CHARTS
# ==============================

if trades:
    df = pd.DataFrame(trades)
    st.subheader("📊 Intraday Setups (Nifty‑50 + Eternal/HDFCLIFE/MARUTI/BOSCH)")

    def color_side(val):
        if val == "LONG":   return 'background-color: #d4f4d4'
        if val == "SHORT":  return 'background-color: #f4d4d4'
        return ''

    st.dataframe(
        df.style
            .applymap(color_side, subset=['Side'])
            .format({
                'Price': '₹{:,.0f}',
                'SL': '₹{:,.0f}',
                'TP': '₹{:,.0f}',
                'Risk ₹': '₹{:,.0f}',
                'Profit ₹': '₹{:,.0f}'
            }),
        use_container_width=True
    )

    # Portfolio summary
    total_risk = df['Risk ₹'].sum()
    total_profit = df['Profit ₹'].sum()
    avg_rr = np.mean([t['Profit ₹'] / max(1, t['Risk ₹']) for _, t in df.iterrows()])

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Capital", f"₹{capital:,}")
    with col2: st.metric("Total Risk", f"₹{total_risk:.0f}", f"{total_risk/capital*100:.1f}%")
    with col3: st.metric("Total Profit", f"₹{total_profit:.0f}")
    with col4: st.metric("Avg R:R", f"{avg_rr:.2f}:1")

    # Charts for top 3 trades
    st.subheader("📈 Live 1‑min Charts (Top Setups)")
    top_trades = df.nlargest(3, 'Risk ₹')
    cols = st.columns(min(3, len(top_trades)))
    for i, (_, trade) in enumerate(top_trades.iterrows()):
        with cols[i]:
            sym = trade['Stock'] + '.NS'
            data = prices_1m[sym]
            if data is None: continue

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index[-30:], open=data['Open'].iloc[-30:],
                high=data['High'].iloc[-30:], low=data['Low'].iloc[-30:],
                close=data['Close'].iloc[-30:], name='Price'
            ))
            fig.add_trace(go.Scatter(x=data.index[-30:], y=data['EMA9'].iloc[-30:], name='EMA9', line=dict(color='lime')))
            fig.add_trace(go.Scatter(x=data.index[-30:], y=data['EMA21'].iloc[-30:], name='EMA21', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=data.index[-30:], y=data['EMA50'].iloc[-30:], name='EMA50', line=dict(color='red')))

            fig.add_hline(y=trade['Price'], line_dash="solid", line_color="blue",
                          annotation_text=f"Entry ₹{trade['Price']:.0f}")
            fig.add_hline(y=trade['SL'], line_dash="dash", line_color="red",
                          annotation_text=f"SL ₹{trade['SL']:.0f}")
            fig.add_hline(y=trade['TP'], line_dash="dash", line_color="green",
                          annotation_text=f"TP ₹{trade['TP']:.0f}")

            st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**🎯 Stocks Included:**

- **Nifty‑50** (subset)  
- **Eternal (Zomato)** → `ETERNAL.NS`  
- **HDFC Life** → `HDFCLIFE.NS`  
- **Maruti** → `MARUTI.NS`  
- **Bosch** → `BOSCHLTD.NS`  

**🔐 Access Control:**

- App is locked behind a **password** (set in Streamlit Secrets when deployed).  
- Only users who know the password can see the dashboard.  

**⚠️ Always paper‑trade first.**
""")
