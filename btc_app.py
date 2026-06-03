# app.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 0) Streamlit page config
st.set_page_config(page_title="Multi-TF Crypto Viewer", layout="wide")

# 1) WMA helper for RSI
def wma(series: pd.Series, window: int) -> pd.Series:
    w = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

# 2) Chunk-fetch base H1 backward via endTime
@st.cache(allow_output_mutation=True)
def fetch_h1(exchange, pair: str, total_bars: int) -> pd.DataFrame:
    limit, left, end_time = 1000, total_bars, None
    chunks = []
    while left > 0:
        cnt = min(limit, left)
        params = {'endTime': end_time} if end_time else {}
        data = exchange.fetch_ohlcv(pair, '1h', limit=cnt, params=params)
        if not data:
            break
        df = pd.DataFrame(data, columns=['ts','open','high','low','close','vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        chunks.insert(0, df)
        left -= len(df)
        end_time = int(df.index.min().timestamp()*1000) - 1
    return pd.concat(chunks).sort_index() if chunks else pd.DataFrame()

# 3) Sidebar controls
st.sidebar.title("⚙️ Settings")
pair     = st.sidebar.selectbox("Pair", ["BTC/USDT", "ETH/USDT"])
tf_list  = ["H1","H2","H4","H6","H8","H12",
            "D1","D2","D3","D4","D5","D6",
            "W1","W2","W3","M1","M2"]
tf_label = st.sidebar.selectbox("Timeframe", tf_list, index=0)
n_bars   = st.sidebar.number_input("Number of bars", 10, 2000, 200, 10)

# 4) Map TF → resample rule & hours-per-bar
tf_map = {
  "H1":("1H",   1), "H2":("2H",   2), "H4":("4H",   4),
  "H6":("6H",   6), "H8":("8H",   8), "H12":("12H",12),
  "D1":("1D",  24), "D2":("2D",  48), "D3":("3D",  72),
  "D4":("4D",  96), "D5":("5D", 120), "D6":("6D", 144),
  "W1":("1W", 168), "W2":("2W", 336), "W3":("3W", 504),
  "M1":("M",  720), "M2":("2M",1440)
}
rule, hrs = tf_map[tf_label]
needed_h1 = hrs * n_bars + 10  # đầu dư thêm vài bars

# 5) Fetch & resample
exchange = ccxt.binance()
df_h1     = fetch_h1(exchange, pair, needed_h1)
df        = (
    df_h1
    .resample(rule)
    .agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'})
    .dropna()
    .tail(n_bars)
)

# 6) Compute RSI(14), EMA9 & WMA45 on RSI
delta    = df['close'].diff()
up       = delta.clip(lower=0)
down     = -delta.clip(upper=0)
ema_up   = up.ewm(alpha=1/14, adjust=False).mean()
ema_dn   = down.ewm(alpha=1/14, adjust=False).mean()
rsi      = 100 - 100 / (1 + ema_up/ema_dn)
ema9_rsi = rsi.ewm(span=9, adjust=False).mean()
wma45    = wma(rsi, 45)

# 7) Volume colors
vol_colors = np.where(df['close'] > df['open'], 'green',
               np.where(df['close'] < df['open'], 'red', 'gray'))

# 8) Build Plotly figure with 3 rows: Price, Volume, RSI
fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.6, 0.2, 0.2],
    shared_xaxes=True,
    vertical_spacing=0.02
)

# 8a) Price candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    ),
    row=1, col=1
)

# 8b) Volume bar
fig.add_trace(
    go.Bar(
        x=df.index, y=df['vol'],
        marker_color=vol_colors,
        showlegend=False
    ),
    row=2, col=1
)

# 8c) RSI + EMA9 + WMA45
fig.add_trace(
    go.Scatter(x=rsi.index,    y=rsi,    line_color='blue',    name='RSI(14)'),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=ema9_rsi.index, y=ema9_rsi, line_color='orange', name='EMA9(RSI)'),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=wma45.index,   y=wma45,   line_color='magenta', name='WMA45(RSI)'),
    row=3, col=1
)

# — Add horizontal lines at 80 & 20 on RSI subplot —
fig.add_hline(y=80, row=3, col=1,
              line_dash="dash", line_color="gray", annotation_text="80", annotation_position="top left")
fig.add_hline(y=20, row=3, col=1,
              line_dash="dash", line_color="gray", annotation_text="20", annotation_position="bottom left")

# 9) Styling: remove grid, white bg, pan mode
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(
    template='plotly_white',
    dragmode='pan',
    title=f"{pair} — {tf_label} — last {n_bars} bars",
    xaxis_rangeslider_visible=False,
    margin=dict(l=20, r=20, t=40, b=20),
    height=750
)

# 10) Chart config: disable scroll zoom, add draw tools
config = {
    'scrollZoom': False,
    'doubleClick': 'reset',
    'modeBarButtonsToAdd': [
        'drawline', 'drawopenpath', 'drawrect',
        'eraseshape', 'drawfibonacci'
    ]
}

# 11) Render
st.plotly_chart(fig, use_container_width=True, config=config)
