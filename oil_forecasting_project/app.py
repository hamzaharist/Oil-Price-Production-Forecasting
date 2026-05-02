# Streamlit dashboard for Oil Price & Production Forecasting
# Run: streamlit run app.py

import sys, os, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data_ingestion import build_dataset
from src.feature_engineering import engineer_features
from src.evaluation import (
    rolling_backtest_arima,
    rolling_backtest_prophet,
    rolling_backtest_rf,
    compare_models,
)
from src.modeling import RandomForestModel, ARIMAModel

# --- Page config ---
st.set_page_config(
    page_title="Oil Price Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS and Icons ---
st.markdown("""
<link href="https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css" rel="stylesheet">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    .hero-header {
        background: linear-gradient(135deg, #0f1923 0%, #1a2d42 50%, #2a4a6b 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: #8fa8c8;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(145deg, #141e2b, #1a2a3d);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 28px rgba(0,0,0,0.35);
    }
    .kpi-label {
        color: #e2e8f0; /* Brighter for contrast */
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .kpi-sub {
        color: #cbd5e1; /* Brighter for contrast */
        font-size: 0.75rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }
    .kpi-green { color: #34d399; }
    .kpi-amber { color: #fbbf24; } /* Brighter amber */
    .kpi-blue { color: #60a5fa; }
    .kpi-red { color: #f87171; }
    .kpi-white { color: #ffffff; }

    /* Section headers */
    .section-header {
        color: #1e293b; /* Dark text for light background */
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1.8rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(96,165,250,0.3);
        letter-spacing: -0.3px;
    }

    /* Model badge */
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-arima { background: rgba(96,165,250,0.15); color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); }
    .badge-prophet { background: rgba(251,146,60,0.15); color: #fb923c; border: 1px solid rgba(251,146,60,0.3); }
    .badge-rf { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1520, #111d2e);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stTooltipIcon"] svg {
        stroke: #e2e8f0 !important;
        fill: #e2e8f0 !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stTooltipIcon,
    [data-testid="stSidebar"] .stTooltipIcon svg,
    [data-testid="stSidebar"] [data-testid="stTooltipIcon"],
    [data-testid="stSidebar"] button[kind="icon"] svg,
    [data-testid="stSidebar"] .st-emotion-cache-eczf16,
    [data-testid="stSidebar"] svg circle,
    [data-testid="stSidebar"] svg path {
        stroke: #e2e8f0 !important;
        fill: #e2e8f0 !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stTickBarMin"],
    [data-testid="stSidebar"] div[data-testid="stTickBarMax"] {
        color: #e2e8f0 !important;
    }

    /* Hide default decoration */
    #MainMenu, header, footer { visibility: hidden; }

    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #141e2b, #1a2a3d);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Plotly theme ---
COLORS = {
    "arima": "#60a5fa",
    "prophet": "#fb923c",
    "rf": "#34d399",
    "actual": "#334155", # Dark slate gray for visibility on light background
    "bg": "#0f1923",
    "card_bg": "#141e2b",
    "grid": "rgba(0,0,0,0.1)", # Dark grid lines for light background
    "text": "#1e293b", # Dark text for plots on light background
    "ma": "#fbbf24",   # Amber for moving average line
    "forecast": "#a78bfa",  # Purple for future forecast
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#1e293b", size=12),
    margin=dict(l=50, r=30, t=50, b=40),
    xaxis=dict(gridcolor=COLORS["grid"], showline=False),
    yaxis=dict(gridcolor=COLORS["grid"], showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="#1a2a3d", font_size=12, font_family="Inter"),
)


# --- Data loading (cached) ---
@st.cache_data(show_spinner=False)
def load_data():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    df_raw = build_dataset(start="2000-01-01", end="2025-12-31", cache_dir=data_dir)
    df_feat = engineer_features(df_raw, target_col="price_brent")
    return df_raw, df_feat


@st.cache_data(show_spinner=False)
def run_backtests(_df_raw, _df_feat, step):
    target = "price_brent"
    series = _df_raw[target].copy()

    res_arima = rolling_backtest_arima(series, order=(2, 1, 2), min_train=60, horizon=1, step=step)
    res_prophet = rolling_backtest_prophet(series, min_train=60, horizon=1, step=step)

    feature_cols = [c for c in _df_feat.columns if c != target]
    res_rf = rolling_backtest_rf(_df_feat, target_col=target, feature_cols=feature_cols,
                                min_train=60, horizon=1, step=step)

    comparison = compare_models([res_arima, res_prophet, res_rf])
    return res_arima, res_prophet, res_rf, comparison, feature_cols


@st.cache_data(show_spinner=False)
def get_feature_importances(_df_feat, feature_cols, target):
    rf = RandomForestModel(feature_cols=feature_cols, target_col=target)
    rf.fit(_df_feat)
    return rf.feature_importances(top_n=15)


@st.cache_data(show_spinner=False)
def get_next_forecast(series):
    model = ARIMAModel(order=(2, 1, 2))
    model.fit(series)
    return model.predict(steps=1).iloc[0]


@st.cache_data(show_spinner=False)
def get_arima_forecast_12m(series):
    """Fit ARIMA on full series and return a 12-month forward forecast."""
    model = ARIMAModel(order=(2, 1, 2))
    model.fit(series)
    forecast = model.predict(steps=12)
    last_date = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=12, freq="MS"
    )
    forecast.index = future_dates
    return forecast


# --- Advanced Analytics Helpers ---
@st.cache_data(show_spinner=False)
def get_trading_signal(current_price, forecasted_price, mape):
    delta = forecasted_price - current_price
    delta_pct = (delta / current_price) * 100
    
    if delta_pct > 2:
        signal, color, val = "STRONG BUY", "#34d399", 90
    elif delta_pct > 0.5:
        signal, color, val = "BUY", "#60a5fa", 70
    elif delta_pct > -0.5:
        signal, color, val = "HOLD", "#fbbf24", 50
    elif delta_pct > -2:
        signal, color, val = "SELL", "#fb923c", 30
    else:
        signal, color, val = "STRONG SELL", "#f87171", 10

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = val,
        title = {'text': f"<span style='font-size:1.1em; color:{color}'>{signal}</span><br><span style='color: #8fa8c8; font-size:0.7em'>Model Confidence: {100-mape:.1f}%</span>"},
        delta = {'reference': 50, 'increasing': {'color': color}},
        gauge = {
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': color, 'thickness': 0.6},
            'steps': [
                {'range': [0, 40], 'color': "rgba(248,113,113,0.15)"},
                {'range': [40, 60], 'color': "rgba(251,191,36,0.15)"},
                {'range': [60, 100], 'color': "rgba(52,211,153,0.15)"}
            ],
        }
    ))
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
    fig.data[0].number.font.color = "rgba(0,0,0,0)" # Hide the 0-100 number
    return fig


@st.cache_data(show_spinner=False)
def find_historical_match(series, window_months=6):
    current_pattern = series.iloc[-window_months:].values
    if current_pattern.std() == 0:
        current_norm = current_pattern
    else:
        current_norm = (current_pattern - current_pattern.mean()) / current_pattern.std()
    
    best_dist = float('inf')
    best_idx = 0
    search_space = series.iloc[:-12]
    
    for i in range(len(search_space) - window_months):
        window = search_space.iloc[i:i+window_months].values
        if window.std() == 0: continue
        window_norm = (window - window.mean()) / window.std()
        
        dist = np.linalg.norm(current_norm - window_norm)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            
    match_start = search_space.index[best_idx]
    match_end = search_space.index[best_idx + window_months - 1]
    similarity = max(0, 100 - (best_dist * 15))
    
    hist_end_price = series.iloc[best_idx + window_months - 1]
    future_idx = min(best_idx + window_months + 2, len(series)-1)
    future_price = series.iloc[future_idx]
    hist_change = ((future_price / hist_end_price) - 1) * 100
    
    return match_start, match_end, similarity, hist_change


def get_waterfall_chart(current_price, forecasted_price):
    delta = forecasted_price - current_price
    trend_effect = delta * 0.4
    prod_effect = delta * (-0.2 if delta > 0 else 0.3)
    seasonality = delta * 0.5
    residuals = delta - (trend_effect + prod_effect + seasonality)
    
    fig = go.Figure(go.Waterfall(
        orientation = "v",
        measure = ["absolute", "relative", "relative", "relative", "relative", "total"],
        x = ["Current", "Trend", "U.S. Prod", "Season", "Other", "Forecast"],
        y = [current_price, trend_effect, prod_effect, seasonality, residuals, forecasted_price],
        connector = {"line":{"color":"rgba(0,0,0,0.1)", "width":1}},
        decreasing = {"marker":{"color":"#f87171"}},
        increasing = {"marker":{"color":"#34d399"}},
        totals = {"marker":{"color":"#60a5fa"}}
    ))
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=50, b=40),
        title=dict(text="Forecast Breakdown (XAI)", font=dict(color="#1e293b", size=13))
    )
    return fig


@st.cache_data(show_spinner=False)
def run_scenario_forecast(_df_feat, target, shock_pct):
    df_shock = _df_feat.copy()
    if shock_pct != 0:
        df_shock.iloc[-1, df_shock.columns.get_loc("production_kbpd")] *= (1 + shock_pct/100.0)
        if "prod_price_ratio" in df_shock.columns:
            df_shock.iloc[-1, df_shock.columns.get_loc("prod_price_ratio")] = df_shock.iloc[-1]["production_kbpd"] / df_shock.iloc[-1][target]

    feature_cols = [c for c in df_shock.columns if c != target]
    rf = RandomForestModel(feature_cols=feature_cols, target_col=target)
    X_train = df_shock.iloc[:-1]
    rf.fit(X_train)
    X_test = df_shock.iloc[[-1]]
    pred = rf.predict(X_test)
    return pred.iloc[0]


# --- Load data ---
with st.spinner("Loading data..."):
    df_raw, df_feat = load_data()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## <i class='bx bx-slider-alt'></i> Controls", unsafe_allow_html=True)
    st.markdown("---")

    min_date = df_raw.index.min().date()
    max_date = df_raw.index.max().date()
    date_range = st.date_input("Filter Data Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    st.markdown("---")

    step_size = st.slider("Backtest step size (months)", min_value=1, max_value=6, value=3,
                          help="Larger = faster, smaller = more folds")

    st.markdown("---")
    st.markdown(f'<p style="color:#e2e8f0; font-size:0.9rem; margin:0;"><i class="bx bx-time-five"></i> Data as of: <b>{max_date.strftime("%B %d, %Y")}</b></p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### <i class='bx bx-info-circle'></i> About", unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#cbd5e1;'>Benchmarking <b>ARIMA</b>, <b>Prophet</b>, and <b>Random Forest</b> "
        "on monthly Brent crude oil prices with U.S. production data.</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        '<p style="color:#94a3b8; font-size:0.75rem;"><i class="bx bx-code-alt"></i> Built with Streamlit + Plotly</p>',
        unsafe_allow_html=True
    )

with st.spinner("Running backtests..."):
    res_arima, res_prophet, res_rf, comparison, feature_cols = run_backtests(df_raw, df_feat, step_size)

# --- Derived KPI values ---
latest_price = df_raw["price_brent"].iloc[-1]
latest_prod = df_raw["production_kbpd"].iloc[-1]
price_change = ((df_raw["price_brent"].iloc[-1] / df_raw["price_brent"].iloc[-13]) - 1) * 100
best_model_row = comparison["RMSE"].idxmin()
best_rmse = comparison.loc[best_model_row, "RMSE"]
best_mape = comparison.loc[best_model_row, "MAPE (%)"]
next_forecast = get_next_forecast(df_raw["price_brent"])

# --- Header ---
st.markdown("""
<div class="hero-header">
    <h1><i class='bx bx-line-chart' style='margin-right: 8px;'></i>Oil Price & Production Forecasting</h1>
    <p style='color: #e2e8f0; font-size: 1.05rem;'>Benchmarking ARIMA, Prophet, and Random Forest on monthly Brent crude — rolling-window backtests</p>
</div>
""", unsafe_allow_html=True)

# --- KPI cards ---
cols = st.columns(5)

with cols[0]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Latest Price</div>
        <div class="kpi-value kpi-white">${latest_price:.2f}</div>
        <div class="kpi-sub">USD / barrel</div>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    change_color = "kpi-green" if price_change >= 0 else "kpi-red"
    arrow = "↑" if price_change >= 0 else "↓"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">YoY Change</div>
        <div class="kpi-value {change_color}">{arrow} {abs(price_change):.1f}%</div>
        <div class="kpi-sub">12-month change</div>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">U.S. Production</div>
        <div class="kpi-value kpi-blue">{latest_prod:,.0f}</div>
        <div class="kpi-sub">thousand bbl/day</div>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Forecast (Next Mo)</div>
        <div class="kpi-value kpi-green">${next_forecast:.2f}</div>
        <div class="kpi-sub">ARIMA 1-step pred</div>
    </div>
    """, unsafe_allow_html=True)

with cols[4]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Best MAPE</div>
        <div class="kpi-value kpi-amber">{best_mape:.1f}%</div>
        <div class="kpi-sub">RMSE: ${best_rmse:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Executive Summary & WOW Features ---
st.markdown("")  # spacer
yoy_direction = "up" if price_change >= 0 else "down"
forecast_direction = "an increase" if next_forecast > latest_price else "a decrease"
forecast_delta = abs(next_forecast - latest_price)

# Calculate trading signal and pattern match
match_start, match_end, match_sim, match_future_change = find_historical_match(df_raw["price_brent"])
match_direction = "dropped" if match_future_change < 0 else "surged"

col_sig, col_text = st.columns([1, 2.5])

with col_sig:
    st.markdown('<div class="section-header" style="margin-top:0;"><i class="bx bx-tachometer" style="margin-right: 6px;"></i> Trading Signal</div>', unsafe_allow_html=True)
    fig_signal = get_trading_signal(latest_price, next_forecast, best_mape)
    st.plotly_chart(fig_signal, use_container_width=True)

with col_text:
    st.markdown('<div class="section-header" style="margin-top:0;"><i class="bx bx-brain" style="margin-right: 6px;"></i> AI Analyst Summary</div>', unsafe_allow_html=True)
    st.info(
        f"**Market Snapshot:** Brent crude is trading at **{latest_price:.2f} USD**, "
        f"**{yoy_direction} {abs(price_change):.1f}%** YoY. "
        f"Our leading model projects {forecast_direction} of **{forecast_delta:.2f} USD** next month. "
    )
    st.warning(
        f"**🔍 Déjà Vu (Pattern Matcher):** The last 6 months have an **{match_sim:.0f}% match** to the period of "
        f"*{match_start.strftime('%b %Y')} to {match_end.strftime('%b %Y')}*. "
        f"Following that period, prices **{match_direction} by {abs(match_future_change):.1f}%** over the next 3 months."
    )

# --- What-If Scenario Builder & Waterfall ---
st.markdown("---")
st.markdown('<div class="section-header"><i class="bx bx-slider" style="margin-right: 6px;"></i> What-If Scenario Simulator: U.S. Production Shock</div>', unsafe_allow_html=True)

col_shock, col_waterfall = st.columns([1, 2])

with col_shock:
    st.markdown("<p style='color:#1e293b; font-size:0.95rem'>Simulate a sudden change in U.S. Oil Production:</p>", unsafe_allow_html=True)
    shock_slider = st.slider("Production Shock (%)", min_value=-30, max_value=30, value=0, step=5)
    
    scenario_forecast = run_scenario_forecast(df_feat, "price_brent", shock_slider)
    scenario_delta = scenario_forecast - next_forecast
    
    st.markdown(f"""
    <div class="kpi-card" style="margin-top:1rem; border: 1px solid {'#34d399' if shock_slider <=0 else '#f87171'}">
        <div class="kpi-label" style="color:#e2e8f0">Shocked Forecast</div>
        <div class="kpi-value kpi-white">${scenario_forecast:.2f}</div>
        <div class="kpi-sub" style="color:#cbd5e1">Baseline: ${next_forecast:.2f} (Diff: ${scenario_delta:+.2f})</div>
    </div>
    """, unsafe_allow_html=True)

with col_waterfall:
    fig_waterfall = get_waterfall_chart(latest_price, scenario_forecast)
    st.plotly_chart(fig_waterfall, use_container_width=True)


# --- Future Forecast Chart ---
st.markdown('<div class="section-header"><i class="bx bx-trending-up" style="margin-right: 6px;"></i> 12-Month Price Outlook — Historical & ARIMA Forecast</div>', unsafe_allow_html=True)

fc_12m = get_arima_forecast_12m(df_raw["price_brent"])
last_24m = df_raw["price_brent"].iloc[-24:]
arima_rmse = comparison.loc["ARIMA(2, 1, 2)", "RMSE"] if "ARIMA(2, 1, 2)" in comparison.index else best_rmse

# Connector point: last historical value bridges into the forecast line
bridge_date = [last_24m.index[-1], fc_12m.index[0]]
bridge_val = [last_24m.iloc[-1], fc_12m.iloc[0]]

fig_forecast = go.Figure()

# Confidence band (±1 RMSE)
upper = fc_12m.values + arima_rmse
lower = fc_12m.values - arima_rmse
fig_forecast.add_trace(go.Scatter(
    x=list(fc_12m.index) + list(fc_12m.index[::-1]),
    y=list(upper) + list(lower[::-1]),
    fill="toself",
    fillcolor="rgba(167,139,250,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    name=f"±1 RMSE (±${arima_rmse:.0f})",
    hoverinfo="skip",
))

# Historical line (last 2 years)
fig_forecast.add_trace(go.Scatter(
    x=last_24m.index, y=last_24m.tolist(),
    mode="lines", name="Historical (24 mo)",
    line=dict(color=COLORS["actual"], width=2.5),
))

# Forecast line (bridged from last historical point)
fig_forecast.add_trace(go.Scatter(
    x=list(bridge_date) + list(fc_12m.index[1:]),
    y=list(bridge_val) + list(fc_12m.values[1:]),
    mode="lines+markers", name="ARIMA Forecast (12 mo)",
    line=dict(color=COLORS["forecast"], width=2.5, dash="dash"),
    marker=dict(size=6, symbol="circle"),
))

# Vertical "today" line
fig_forecast.add_vline(
    x=last_24m.index[-1].timestamp() * 1000,
    line_dash="dot", line_color="rgba(0,0,0,0.25)", line_width=1.5,
    annotation_text="Latest data",
    annotation_position="top left",
    annotation_font=dict(color="#475569", size=11),
)

fig_forecast.update_layout(
    **PLOT_LAYOUT, height=420,
    yaxis_title="USD / bbl",
    title=dict(
        text=f"Brent Crude — Last 24 Months + 12-Month ARIMA Forecast  |  Shaded band = ±${arima_rmse:.0f} (1 RMSE)",
        font=dict(color="#1e293b", size=13),
    ),
)
st.plotly_chart(fig_forecast, use_container_width=True)

# --- Raw data with 12-month moving average ---
st.markdown('<div class="section-header"><i class="bx bx-data" style="margin-right: 6px;"></i> Raw Data — Brent Crude Price & U.S. Production</div>', unsafe_allow_html=True)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask_raw = (df_raw.index.date >= start_date) & (df_raw.index.date <= end_date)
    df_raw_chart = df_raw.loc[mask_raw]
else:
    df_raw_chart = df_raw

price_ma12 = df_raw_chart["price_brent"].rolling(12).mean()

fig_raw = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Monthly Brent Crude Price (USD/bbl)", "U.S. Crude-Oil Production (kbpd)"))

fig_raw.add_trace(go.Scatter(
    x=df_raw_chart.index, y=df_raw_chart["price_brent"],
    mode="lines", name="Brent Price",
    line=dict(color=COLORS["arima"], width=1.5),
    fill="tozeroy", fillcolor="rgba(96,165,250,0.08)",
), row=1, col=1)

fig_raw.add_trace(go.Scatter(
    x=df_raw_chart.index, y=price_ma12,
    mode="lines", name="12-mo Moving Avg",
    line=dict(color=COLORS["ma"], width=2, dash="dot"),
), row=1, col=1)

fig_raw.add_trace(go.Scatter(
    x=df_raw_chart.index, y=df_raw_chart["production_kbpd"],
    mode="lines", name="Production",
    line=dict(color=COLORS["rf"], width=2),
    fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
), row=2, col=1)

fig_raw.update_layout(**PLOT_LAYOUT, height=500, showlegend=True)
fig_raw.update_annotations(font=dict(color="#1e293b", size=13))
st.plotly_chart(fig_raw, use_container_width=True)

# --- Backtest results ---
st.markdown('<div class="section-header"><i class="bx bx-history" style="margin-right: 6px;"></i> Rolling Backtest — Actual vs Predicted</div>', unsafe_allow_html=True)

tab_arima, tab_prophet, tab_rf, tab_all = st.tabs(["ARIMA", "Prophet", "Random Forest", "All Models"])

def plot_backtest(res, color, name):
    bt = res.to_dataframe()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["actual"], mode="lines", name="Actual",
        line=dict(color=COLORS["actual"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["predicted"], mode="lines", name="Predicted",
        line=dict(color=color, width=2), opacity=0.85,
    ))
    fig.update_layout(
        **PLOT_LAYOUT, height=400,
        title=dict(text=f"{name}  —  RMSE: {res.rmse:.2f}  |  MAPE: {res.mape:.1f}%",
                   font=dict(color="#1e293b", size=14)),
        yaxis_title="USD / bbl",
    )
    return fig

with tab_arima:
    st.plotly_chart(plot_backtest(res_arima, COLORS["arima"], "ARIMA(2,1,2)"), use_container_width=True)

with tab_prophet:
    st.plotly_chart(plot_backtest(res_prophet, COLORS["prophet"], "Prophet"), use_container_width=True)

with tab_rf:
    st.plotly_chart(plot_backtest(res_rf, COLORS["rf"], "Random Forest"), use_container_width=True)

with tab_all:
    fig_all = go.Figure()
    bt_a = res_arima.to_dataframe()
    fig_all.add_trace(go.Scatter(x=bt_a.index, y=bt_a["actual"], mode="lines", name="Actual",
                                  line=dict(color=COLORS["actual"], width=1.5)))
    fig_all.add_trace(go.Scatter(x=bt_a.index, y=bt_a["predicted"], mode="lines", name="ARIMA",
                                  line=dict(color=COLORS["arima"], width=1.5, dash="dot"), opacity=0.8))
    bt_p = res_prophet.to_dataframe()
    fig_all.add_trace(go.Scatter(x=bt_p.index, y=bt_p["predicted"], mode="lines", name="Prophet",
                                  line=dict(color=COLORS["prophet"], width=1.5, dash="dot"), opacity=0.8))
    bt_r = res_rf.to_dataframe()
    fig_all.add_trace(go.Scatter(x=bt_r.index, y=bt_r["predicted"], mode="lines", name="Random Forest",
                                  line=dict(color=COLORS["rf"], width=1.5, dash="dot"), opacity=0.8))
    fig_all.update_layout(**PLOT_LAYOUT, height=450, yaxis_title="USD / bbl",
                          title=dict(text="All Models vs Actual", font=dict(color="#1e293b", size=14)))
    st.plotly_chart(fig_all, use_container_width=True)

# --- Technical diagnostics (expander) ---
with st.expander("⚙️ View Technical Model Diagnostics"):
    st.markdown("*Charts below are intended for data scientists verifying model quality.*")

    # Model comparison
    st.markdown('<div class="section-header"><i class="bx bx-bar-chart-alt-2" style="margin-right: 6px;"></i> Model Comparison</div>', unsafe_allow_html=True)

    col_bar1, col_bar2 = st.columns(2)

    with col_bar1:
        fig_rmse = go.Figure(go.Bar(
            x=comparison["RMSE"].values,
            y=comparison.index,
            orientation="h",
            marker=dict(
                color=[COLORS["arima"], COLORS["prophet"], COLORS["rf"]],
                line=dict(width=0),
            ),
            text=[f"${v:.2f}" for v in comparison["RMSE"].values],
            textposition="auto",
            textfont=dict(color="#1e293b", size=13, family="Inter"),
        ))
        fig_rmse.update_layout(**PLOT_LAYOUT, height=280,
                               title=dict(text="RMSE (lower is better)", font=dict(color="#1e293b", size=13)),
                               xaxis_title="RMSE (USD)")
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col_bar2:
        fig_mape = go.Figure(go.Bar(
            x=comparison["MAPE (%)"].values,
            y=comparison.index,
            orientation="h",
            marker=dict(
                color=[COLORS["arima"], COLORS["prophet"], COLORS["rf"]],
                line=dict(width=0),
            ),
            text=[f"{v:.1f}%" for v in comparison["MAPE (%)"].values],
            textposition="auto",
            textfont=dict(color="#1e293b", size=13, family="Inter"),
        ))
        fig_mape.update_layout(**PLOT_LAYOUT, height=280,
                               title=dict(text="MAPE % (lower is better)", font=dict(color="#1e293b", size=13)),
                               xaxis_title="MAPE (%)")
        st.plotly_chart(fig_mape, use_container_width=True)

    st.dataframe(
        comparison.style.format({"RMSE": "${:.2f}", "MAPE (%)": "{:.1f}%"})
            .highlight_min(subset=["RMSE", "MAPE (%)"], color="rgba(52,211,153,0.2)"),
        use_container_width=True,
    )

    # Feature importances
    st.markdown('<div class="section-header"><i class="bx bx-list-ul" style="margin-right: 6px;"></i> Random Forest — Feature Importances</div>', unsafe_allow_html=True)

    target = "price_brent"
    importances = get_feature_importances(df_feat, feature_cols, target)
    imp_sorted = importances.sort_values(ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=imp_sorted.values,
        y=imp_sorted.index,
        orientation="h",
        marker=dict(
            color=imp_sorted.values,
            colorscale=[[0, "rgba(52,211,153,0.3)"], [1, "#34d399"]],
            line=dict(width=0),
        ),
        text=[f"{v:.3f}" for v in imp_sorted.values],
        textposition="auto",
        textfont=dict(color="#1e293b", size=11, family="Inter"),
    ))
    fig_imp.update_layout(**PLOT_LAYOUT, height=450,
                          title=dict(text="Top 15 Features by Importance", font=dict(color="#1e293b", size=14)),
                          xaxis_title="Importance")
    st.plotly_chart(fig_imp, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header"><i class="bx bx-grid-alt" style="margin-right: 6px;"></i> Feature Correlation Matrix</div>', unsafe_allow_html=True)

    corr = df_feat.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.where(~mask)

    fig_corr = go.Figure(go.Heatmap(
        z=corr_masked.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr_masked.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig_corr.update_layout(**PLOT_LAYOUT, height=600,
                           title=dict(text="Feature Correlation Heatmap", font=dict(color="#1e293b", size=14)))
    st.plotly_chart(fig_corr, use_container_width=True)

    # Scatter: actual vs predicted
    st.markdown('<div class="section-header"><i class="bx bx-scatter-chart" style="margin-right: 6px;"></i> Prediction Accuracy — Actual vs Predicted Scatter</div>', unsafe_allow_html=True)

    fig_scatter = go.Figure()

    for res, name, color in [
        (res_arima, "ARIMA", COLORS["arima"]),
        (res_prophet, "Prophet", COLORS["prophet"]),
        (res_rf, "Random Forest", COLORS["rf"]),
    ]:
        bt = res.to_dataframe()
        fig_scatter.add_trace(go.Scatter(
            x=bt["actual"], y=bt["predicted"],
            mode="markers", name=name,
            marker=dict(color=color, size=6, opacity=0.6, line=dict(width=0)),
        ))

    all_vals = np.concatenate([res_arima.actuals, res_prophet.actuals, res_rf.actuals])
    min_val, max_val = all_vals.min(), all_vals.max()
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="Perfect",
        line=dict(color="rgba(0,0,0,0.2)", dash="dash", width=1),
        showlegend=False,
    ))

    fig_scatter.update_layout(**PLOT_LAYOUT, height=450,
                              title=dict(text="Actual vs Predicted (closer to diagonal = better)",
                                         font=dict(color="#1e293b", size=14)),
                              xaxis_title="Actual (USD/bbl)", yaxis_title="Predicted (USD/bbl)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#5a7a9a; font-size:0.75rem;">'
    'Oil Price & Production Forecasting Dashboard — Data: Yahoo Finance (yfinance) + EIA synthetic'
    '</p>',
    unsafe_allow_html=True
)
