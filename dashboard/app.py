"""
dashboard/app.py  —  Real Time Fraud Detection System
Single-page batch fraud detection dashboard.

Run:
    streamlit run dashboard/app.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from predict      import predict_batch
from shap_explain import get_top_features

# ── Friendly labels ────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "TransactionAmt" : "Transaction Amount ($)",
    "card4"          : "Card Brand",
    "card6"          : "Card Type",
    "P_emaildomain"  : "Purchaser Email Domain",
    "addr1"          : "Billing ZIP Code",
    "addr2"          : "Billing Country Code",
    "M4"             : "Bank Match Score",
    "M5"             : "Email Domain Match",
    "M6"             : "Billing Bank Match",
    "V70"            : "Purchase Velocity Score",
    "V91"            : "Card Activity Score",
    "V317"           : "Fraud Pattern Score",
    "card1"          : "Card Number Prefix",
    "card2"          : "Card BIN Group",
    "card3"          : "Card Issuing Region",
    "card5"          : "Card Bank Code",
    "ProductCD"      : "Product Category",
    "dist1"          : "Distance: Card to Billing Address",
    "C1"  : "No. of Cards on Billing Address",
    "C2"  : "No. of Linked Bank Accounts",
    "C4"  : "Transactions with Same Card",
    "C6"  : "Billing Addresses on Card",
    "C8"  : "Unique Emails on Card",
    "C11" : "Accounts Sharing Card",
    "C13" : "Emails Linked to Card",
    "C14" : "Cards on Same Account",
    "D1"  : "Days Since Last Transaction",
    "D10" : "Days Since Last Purchase",
    "D11" : "Days Since Account Creation",
    "D15" : "Days Since Last Activity",
    "M1"  : "Card Name Match",
    "M2"  : "Billing Address Match",
    "M3"  : "Phone Number Match",
    "V95"  : "Transaction Frequency Score",
    "V258" : "Network Behavior Score",
    "V283" : "Cross-Channel Activity Score",
    "V308" : "Risk Signal B",
    "V133" : "Session Activity Score",
    "C8"   : "Unique Emails on Card",
    "DeviceType" : "Device Type",
    "DeviceInfo" : "Device Model",
    "id_30": "Operating System",
    "id_31": "Browser Type",
}

BATCH_COL_LABELS = {
    "TransactionID"     : "Transaction ID",
    "TransactionAmt"    : "Amount ($)",
    "ProductCD"         : "Product",
    "card4"             : "Card Brand",
    "card6"             : "Card Type",
    "fraud_probability" : "Fraud Probability",
    "fraud_predicted"   : "Flagged",
    "risk_tier"         : "Risk Tier",
}

PRODUCT_MAP = {"W":"Web Purchase","H":"Hotel / Travel","C":"Cash","S":"Services","R":"Retail"}
TIER_CLR    = {"Critical":"#ff4560","High":"#ffa500","Medium":"#ffdd00","Low":"#00e396"}

def friendly(name):
    return FEATURE_LABELS.get(name, name)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Real Time Fraud Detection System", page_icon="🔍",
                   layout="wide", initial_sidebar_state="collapsed")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root{--bg:#0a0e1a;--surface:#111827;--border:#1e2d40;--accent:#00d4ff;
      --danger:#ff4560;--warn:#ffa500;--ok:#00e396;--text:#e2e8f0;--muted:#64748b;--card:#0f172a;}

html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stHeader"]{background:transparent!important;}
h1,h2,h3{font-family:'Space Mono',monospace!important;}

.fl-header{padding:2.5rem 0 1.5rem;border-bottom:1px solid var(--border);margin-bottom:2rem;}
.fl-title{font-family:'Space Mono',monospace;font-size:2.2rem;font-weight:700;color:var(--accent);letter-spacing:-1px;margin:0;}
.fl-sub{color:var(--muted);font-size:.95rem;margin-top:.3rem;}

.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1.5rem;text-align:center;margin-top:.5rem;}
.card-val{font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;line-height:1;}
.card-lbl{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:.4rem;}

.badge{display:inline-block;padding:.35rem 1rem;border-radius:999px;font-family:'Space Mono',monospace;font-size:.8rem;font-weight:700;letter-spacing:1px;}
.badge-critical{background:#ff456022;color:#ff4560;border:1px solid #ff4560;}
.badge-high{background:#ffa50022;color:#ffa500;border:1px solid #ffa500;}
.badge-medium{background:#ffdd0022;color:#ffdd00;border:1px solid #ffdd00;}
.badge-low{background:#00e39622;color:#00e396;border:1px solid #00e396;}

.sec{font-family:'Space Mono',monospace;font-size:.75rem;text-transform:uppercase;
     letter-spacing:2px;color:var(--muted);border-left:3px solid var(--accent);
     padding-left:.6rem;margin:1.8rem 0 1rem;}

.upload-hint{background:var(--surface);border:1px dashed var(--border);border-radius:12px;
             padding:2rem;text-align:center;color:var(--muted);font-size:.9rem;}

[data-testid="stButton"]>button{background:var(--accent)!important;color:#0a0e1a!important;
    font-family:'Space Mono',monospace!important;font-weight:700!important;border:none!important;
    border-radius:8px!important;padding:.6rem 2rem!important;font-size:.9rem!important;letter-spacing:1px!important;}
[data-testid="stButton"]>button:hover{opacity:.85!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:10px!important;}
hr{border-color:var(--border)!important;}
[data-testid="stFileUploader"]{background:var(--surface)!important;border-radius:10px!important;}

/* expander */
[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important;}
</style>
""", unsafe_allow_html=True)

# ── Chart helpers ──────────────────────────────────────────────────────────────
BG="#0f172a"; GRD="#1e2d40"; TXT="#e2e8f0"
ACC="#00d4ff"; DNG="#ff4560"; WRN="#ffa500"; OKC="#00e396"

def _layout(fig, title=""):
    fig.update_layout(title=title, paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(color=TXT,family="DM Sans"),
                      margin=dict(l=20,r=20,t=40 if title else 20,b=20),
                      xaxis=dict(gridcolor=GRD,zerolinecolor=GRD),
                      yaxis=dict(gridcolor=GRD,zerolinecolor=GRD),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def badge(tier):
    return f'<span class="badge badge-{tier.lower()}">{tier}</span>'

def donut(df):
    counts = df["risk_tier"].value_counts().reindex(
        ["Critical","High","Medium","Low"], fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=.65,
        marker_colors=[TIER_CLR[t] for t in counts.index],
        textinfo="label+percent", textfont=dict(size=12,color=TXT),
    ))
    _layout(fig,"Risk Tier Breakdown")
    fig.update_layout(height=320, showlegend=False)
    return fig

def histogram(df):
    fig = go.Figure(go.Histogram(
        x=df["fraud_probability"], nbinsx=50, marker_color=ACC, opacity=.8))
    _layout(fig,"Fraud Probability Distribution")
    fig.update_layout(height=320, xaxis_title="Fraud Probability",
                      yaxis_title="No. of Transactions", bargap=.05)
    fig.add_vline(x=.5, line_dash="dash", line_color=WRN,
                  annotation_text="Threshold", annotation_font_color=WRN)
    return fig

def amount_by_tier(df):
    grp = df.groupby("risk_tier")["TransactionAmt"].sum().reindex(
        ["Critical","High","Medium","Low"], fill_value=0)
    fig = go.Figure(go.Bar(
        x=grp.index, y=grp.values,
        marker_color=[TIER_CLR[t] for t in grp.index],
        text=[f"${v:,.0f}" for v in grp.values],
        textposition="outside", textfont=dict(color=TXT),
    ))
    _layout(fig,"Total Transaction Value by Risk Tier")
    fig.update_layout(height=320, yaxis_title="Total Amount ($)",
                      xaxis_title="Risk Tier", showlegend=False)
    return fig

def shap_chart(features):
    top    = features[:10]
    labels = [friendly(f["feature"]) for f in top]
    values = [f["shap"] for f in top]
    colors = [DNG if v > 0 else OKC for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside", textfont=dict(size=11,color=TXT),
    ))
    _layout(fig, "What drove this prediction?")
    fig.update_layout(height=400,
                      yaxis=dict(autorange="reversed", gridcolor=GRD),
                      xaxis_title="Impact on fraud score  (red = increases risk · green = decreases risk)")
    fig.add_vline(x=0, line_width=1, line_color=TXT, opacity=.3)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="fl-header">
  <p class="fl-title">Real-Time Fraud Detection System</p>
  <p class="fl-sub">ML · XGBoost + SHAP · IEEE-CIS Dataset</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  UPLOAD SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="sec">Upload Transaction File</p>', unsafe_allow_html=True)

uc, cc = st.columns([2, 1])
with uc:
    uploaded = st.file_uploader(
        "Drop your transaction CSV here  (same format as train_transaction.csv)",
        type=["csv"],
        help="Upload a raw transaction file — all feature columns will be used for prediction.")
with cc:
    threshold = st.slider("Fraud Decision Threshold", 0.0, 1.0, 0.5, 0.01,
                          help="Transactions above this probability are flagged as fraud")
    max_rows  = st.number_input("Max transactions to score",
                                min_value=100, max_value=100000, value=5000, step=500)
    show_n    = st.number_input("Rows to display in table",
                                min_value=10,  max_value=500,    value=50,   step=10)

if not uploaded:
    st.markdown("""
    <div class="upload-hint">
        📂  Upload a <b>transaction CSV</b> above to get started.<br><br>
        The file should follow the same schema as <code>train_transaction.csv</code><br>
        from the IEEE-CIS Fraud Detection dataset.
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading file…"):
    df_raw = pd.read_csv(uploaded, low_memory=False, nrows=int(max_rows))
st.success(f" Loaded **{len(df_raw):,} transactions** · **{df_raw.shape[1]} columns**")

if st.button(" RUN FRAUD DETECTION"):
    with st.spinner(f"Scoring {len(df_raw):,} transactions — this may take a moment…"):
        try:
            result_df = predict_batch(df_raw, threshold=threshold)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()
    st.session_state["result"] = result_df
    st.session_state["raw"]    = df_raw
    

    # ── MLOps Logging ────────────────────────────────────────────────
    try:
        log_dir = os.path.join(ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_df = result_df.copy()
        log_df["timestamp"] = datetime.now()

        log_path = os.path.join(log_dir, "predictions_log.csv")

        write_header = not os.path.exists(log_path)

        log_df.to_csv(log_path, mode='a', header=write_header, index=False)

    except Exception as e:
        st.warning(f"Logging failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if "result" not in st.session_state:
    st.stop()

res = st.session_state["result"]
raw = st.session_state["raw"]

total   = len(res)
n_fraud = int(res["fraud_predicted"].sum())
n_crit  = int((res["risk_tier"] == "Critical").sum())
n_high  = int((res["risk_tier"] == "High").sum())
avg_p   = res["fraud_probability"].mean()
total_amt_fraud = res[res["fraud_predicted"]==1]["TransactionAmt"].sum() \
    if "TransactionAmt" in res.columns else 0

# ── KPI cards ──────────────────────────────────────────────────────────────────
st.markdown('<p class="sec">Detection Summary</p>', unsafe_allow_html=True)

k1,k2,k3,k4,k5 = st.columns(5)
for col, val, lbl, clr in [
    (k1, f"{total:,}",              "Transactions Scored",   TXT),
    (k2, f"{n_fraud:,}",            "Flagged as Fraud",      DNG),
    (k3, f"{n_crit:,}",             "Critical Alerts",       WRN),
    (k4, f"{avg_p*100:.1f}%",       "Avg Fraud Probability", ACC),
    (k5, f"${total_amt_fraud:,.0f}","At-Risk Amount ($)",    DNG),
]:
    with col:
        st.markdown(f"""<div class="card">
            <div class="card-val" style="color:{clr}">{val}</div>
            <div class="card-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec">Visual Breakdown</p>', unsafe_allow_html=True)

v1, v2, v3 = st.columns(3)
with v1: st.plotly_chart(donut(res),     use_container_width=True)
with v2: st.plotly_chart(histogram(res), use_container_width=True)
with v3:
    if "TransactionAmt" in res.columns:
        st.plotly_chart(amount_by_tier(res), use_container_width=True)
    else:
        st.info("TransactionAmt column not found for amount chart.")

# ── Top flagged transactions table ─────────────────────────────────────────────
st.markdown('<p class="sec">Top Flagged Transactions</p>', unsafe_allow_html=True)

raw_cols = [c for c in ["TransactionID","TransactionAmt","ProductCD",
                         "card4","card6","fraud_probability","fraud_predicted","risk_tier"]
            if c in res.columns]
top = (res[res["fraud_predicted"]==1]
       .nlargest(int(show_n), "fraud_probability")[raw_cols])
disp = top.rename(columns=BATCH_COL_LABELS)
if "Product" in disp.columns:
    disp["Product"] = disp["Product"].map(PRODUCT_MAP).fillna(disp["Product"])

st.dataframe(
    disp.style
        .background_gradient(subset=["Fraud Probability"], cmap="Reds")
        .format({"Fraud Probability": "{:.4f}"}),
    use_container_width=True, height=400)

# ── SHAP drill-down ────────────────────────────────────────────────────────────
st.markdown('<p class="sec">Transaction Deep-Dive — SHAP Explanation</p>',
            unsafe_allow_html=True)

flagged_ids = res[res["fraud_predicted"]==1].nlargest(50,"fraud_probability")

if "TransactionID" in flagged_ids.columns:
    id_options = flagged_ids["TransactionID"].astype(str).tolist()
    sel_id = st.selectbox(
        "Select a flagged Transaction ID to explain:",
        options=id_options,
        format_func=lambda x: f"Transaction {x}  —  "
            f"Prob: {flagged_ids[flagged_ids['TransactionID'].astype(str)==x]['fraud_probability'].values[0]:.4f}  "
            f"| {flagged_ids[flagged_ids['TransactionID'].astype(str)==x]['risk_tier'].values[0]}")

    if st.button(" EXPLAIN THIS TRANSACTION"):
        sel_row = raw[raw["TransactionID"].astype(str)==sel_id].iloc[0].to_dict()
        pred_row = res[res["TransactionID"].astype(str)==sel_id].iloc[0]

        with st.spinner("Computing SHAP values…"):
            try:
                features = get_top_features(sel_row, top_n=12)
            except Exception as e:
                st.error(f"SHAP error: {e}"); st.stop()

        prob = pred_row["fraud_probability"]
        tier = pred_row["risk_tier"]

        sc, ic = st.columns([2, 1])
        with sc:
            st.plotly_chart(shap_chart(features), use_container_width=True)
        with ic:
            st.markdown(f"""
            <div class="card">
                <div class="card-val" style="color:{TIER_CLR[tier]}">{prob*100:.1f}%</div>
                <div class="card-lbl">Fraud Probability</div>
            </div>
            <div style="text-align:center;margin-top:.8rem">{badge(tier)}</div>
            """, unsafe_allow_html=True)
            st.markdown("**Key risk factors:**")
            for f in features[:6]:
                icon = "🔴" if f["direction"]=="increases_risk" else "🟢"
                st.markdown(f"{icon} **{friendly(f['feature'])}**  \n"
                            f"Value: `{f['value']:.2f}` · Impact: `{f['shap']:+.4f}`")
else:
    st.info("TransactionID column not found — cannot drill into individual transactions.")

# ── Export ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec">Export</p>', unsafe_allow_html=True)
st.download_button(
    " Download Full Predictions CSV",
    data=res.to_csv(index=False).encode("utf-8"),
    file_name="fraud_predictions.csv",
    mime="text/csv")