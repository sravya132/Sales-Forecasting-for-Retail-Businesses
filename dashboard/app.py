import streamlit as st
import pandas as pd
import plotly.express as px
import os

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# =============================
# CUSTOM CSS (UNIQUE UI)
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
}

.dashboard-title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.dashboard-subtitle {
    text-align: center;
    font-size: 18px;
    color: #dcdcdc;
    margin-bottom: 40px;
}

.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-top: 40px;
    margin-bottom: 20px;
    color: #00d9f5;
}

.kpi-card {
    background: rgba(255, 255, 255, 0.14);
    backdrop-filter: blur(16px);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

.kpi-title {
    font-size: 16px;
    color: #cccccc;
}

.kpi-value {
    font-size: 38px;
    font-weight: 700;
    color: #00f5a0;
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD DATA (ROBUST PATH)
# =============================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "forecast_output.csv")
    return pd.read_csv(data_path)

df = load_data()
df["Date"] = pd.to_datetime(df["Date"])

# =============================
# TITLE
# =============================
st.markdown("""
<div class="dashboard-title">📊 Retail Sales Forecasting</div>
<div class="dashboard-subtitle">
AI-powered dashboard for analyzing and forecasting retail sales
</div>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR FILTERS
# =============================
st.sidebar.header("📅 Date Filter")

start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
]

# =============================
# KPI SECTION
# =============================
st.markdown('<div class="section-title">📌 Key Performance Indicators</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Actual Sales</div>
        <div class="kpi-value">{int(filtered_df["Actual_Sales"].sum())}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Predicted Sales</div>
        <div class="kpi-value">{int(filtered_df["Predicted_Sales"].sum())}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average Daily Sales</div>
        <div class="kpi-value">{int(filtered_df["Actual_Sales"].mean())}</div>
    </div>
    """, unsafe_allow_html=True)

# =============================
# GRAPH 1: ACTUAL VS PREDICTED
# =============================
st.markdown('<div class="section-title">📈 Actual vs Forecasted Sales</div>', unsafe_allow_html=True)

fig1 = px.line(
    filtered_df,
    x="Date",
    y=["Actual_Sales", "Predicted_Sales"]
)

fig1.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified"
)

st.plotly_chart(fig1, use_container_width=True)

# =============================
# GRAPH 2: SALES DISTRIBUTION
# =============================
st.markdown('<div class="section-title">📊 Sales Distribution</div>', unsafe_allow_html=True)

fig2 = px.histogram(
    filtered_df,
    x="Actual_Sales",
    nbins=30
)

fig2.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig2, use_container_width=True)

# =============================
# GRAPH 3: FORECAST ERROR (RESIDUALS)
# =============================
filtered_df["Residual"] = filtered_df["Actual_Sales"] - filtered_df["Predicted_Sales"]

st.markdown('<div class="section-title">📉 Forecast Error Analysis</div>', unsafe_allow_html=True)

fig3 = px.line(
    filtered_df,
    x="Date",
    y="Residual"
)

fig3.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified"
)

st.plotly_chart(fig3, use_container_width=True)

# =============================
# DATA TABLE
# =============================
st.markdown('<div class="section-title">📋 Forecast Data</div>', unsafe_allow_html=True)
st.dataframe(filtered_df, use_container_width=True)

# =============================
# DOWNLOAD BUTTON
# =============================
st.download_button(
    "⬇️ Download Forecast Data",
    filtered_df.to_csv(index=False),
    file_name="forecast_results.csv",
    mime="text/csv"
)

# =============================
# FOOTER
# =============================
st.markdown("""
<hr>
<p style="text-align:center; font-size:14px; color:#cccccc;">
Big Data Analytics Project | Sales Forecasting Dashboard
</p>
""", unsafe_allow_html=True)
