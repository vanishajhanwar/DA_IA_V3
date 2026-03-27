"""
app.py — Warehouse Rental Management Analytics Platform
Data Analytics MGB Project | Family Business Use Case
Run: streamlit run app.py
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from notifications import render_notification_page
from settings import render_settings_page, init_settings
from admin import render_admin_page
from data_manager import get_transaction_df, get_business
from utils import (
    load_data, apply_filters, compute_kpis, inr_fmt,
    train_classifier, train_rent_regressor, train_delay_regressor,
    train_clustering, compute_association_rules, score_prospects,
    COLORS, PALETTE, PROSPECT_SCHEMA,
)
from charts import (
    monthly_revenue_trend, owner_revenue_bar, owner_revenue_pie,
    location_bar, payment_status_donut, payment_behaviour_bar,
    quarterly_revenue, wh_type_revenue, automation_summary,
    correlation_heatmap, rent_vs_size, rent_vs_type,
    delay_by_industry, delay_by_tenant_type, risk_score_histogram,
    location_type_heatmap,
    roc_curve_chart, confusion_matrix_chart, feature_importance_chart,
    regression_actual_vs_predicted,
    elbow_chart, cluster_scatter, cluster_profile_radar,
    association_heatmap, association_bubble,
    prospect_gauge, prospect_risk_bar,
)

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WareHub Analytics",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F1923;
    border-right: 1px solid #1E2D3D;
}
[data-testid="stSidebar"] * { color: #C8D8E8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] h1, h2, h3 { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #90A4AE !important; font-size: 12px; }

/* KPI cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5E0D8;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(15,25,35,.07);
    transition: box-shadow .2s;
}
.kpi-card:hover { box-shadow: 0 4px 16px rgba(15,25,35,.12); }
.kpi-label { font-size: 11px; font-weight: 600; color: #64748B;
             text-transform: uppercase; letter-spacing: .07em; margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 700; color: #0F1923;
             letter-spacing: -.03em; line-height: 1; }
.kpi-footer { font-size: 11px; color: #94A3B8; margin-top: 6px; }
.kpi-tag { display: inline-block; font-size: 10px; font-weight: 600;
           padding: 2px 9px; border-radius: 10px; margin-top: 8px;
           text-transform: uppercase; letter-spacing: .04em; }
.tag-green  { background: #E8F6F0; color: #1D8A5F; }
.tag-red    { background: #FCEAEA; color: #C0392B; }
.tag-amber  { background: #FDF6E3; color: #D97706; }
.tag-blue   { background: #EBF2FA; color: #1A3C5E; }

/* Alert */
.alert-danger {
    background: #FCEAEA; border: 1px solid #F5C4C4;
    border-radius: 10px; padding: 12px 16px; margin-bottom: 16px;
    font-size: 13px; color: #C0392B;
}
.alert-info {
    background: #EBF2FA; border: 1px solid #BDD5EE;
    border-radius: 10px; padding: 12px 16px; margin-bottom: 16px;
    font-size: 13px; color: #1A3C5E;
}
.alert-success {
    background: #E8F6F0; border: 1px solid #A8DDD8;
    border-radius: 10px; padding: 12px 16px; margin-bottom: 16px;
    font-size: 13px; color: #1D8A5F;
}

/* Section header */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 22px; color: #0F1923;
    letter-spacing: -.02em; margin-bottom: 4px;
}
.section-sub { font-size: 13px; color: #64748B; margin-bottom: 20px; }

/* Metric badge table */
.metric-badge {
    display: inline-block; padding: 3px 10px; border-radius: 10px;
    font-size: 11px; font-weight: 600;
}

/* Divider */
.divider { border: none; border-top: 1px solid #E5E0D8; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def get_data():
    return load_data("warehouse_data.csv")

df_main = get_transaction_df()
init_settings()

# Pre-fill WhatsApp number and email from business profile
biz = get_business()
if biz.get("phone") and not st.session_state.get("sender_phone_set"):
    import re as _re
    ph = _re.sub(r"\D","", biz["phone"])
    if not ph.startswith("91"): ph = "91" + ph
    st.session_state["sender_phone"] = ph
    st.session_state["sender_phone_set"] = True
if biz.get("email") and not st.session_state.get("gmail_user"):
    st.session_state["gmail_user_prefill"] = biz["email"]


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 WareHub Analytics")
    st.markdown("Warehouse Rental Management\nData Analytics MGB Project")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=[
            "📊 Overview",
            "💰 Revenue Analysis",
            "🔍 Diagnostic Analysis",
            "🤖 Predictive — Classification",
            "📈 Predictive — Regression",
            "🎯 Clustering",
            "🔗 Association Rules",
            "🚀 Prospect Scoring",
            "📨 Invoice & Reminders",
            "📋 Data Explorer",
            "⚙️ Settings",
            "🗂️ Admin — Manage Data",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Global Filters**")

    owners    = ["All"] + sorted(df_main["Owner_Name"].dropna().unique().tolist())
    locations = ["All"] + sorted(df_main["Warehouse_Location"].dropna().unique().tolist())
    wh_types  = ["All"] + sorted(df_main["Warehouse_Type"].dropna().unique().tolist())
    years     = ["All"] + sorted(df_main["Month"].dt.year.dropna().unique().astype(str).tolist())

    sel_owner    = st.selectbox("Owner",    owners)
    sel_location = st.selectbox("Location", locations)
    sel_type     = st.selectbox("WH Type",  wh_types)
    sel_year     = st.selectbox("Year",     years)
    sel_pay      = st.selectbox("Payment",  ["All", "Paid", "Not Paid"])

    df = apply_filters(
        df_main,
        owner=sel_owner, location=sel_location,
        wh_type=sel_type, pay_status=sel_pay, year=sel_year,
    )

    st.markdown("---")
    n_filtered = len(df)
    n_total    = len(df_main)
    st.markdown(f"**Showing:** {n_filtered} / {n_total} records")
    if n_filtered < n_total:
        st.markdown(
            f'<div style="background:#EBF2FA;padding:6px 10px;border-radius:6px;'
            f'font-size:11px;color:#1A3C5E">🔍 {n_total-n_filtered} records filtered out</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Dataset Stats**")
    from data_manager import get_tenants as _get_tnt, get_warehouses as _get_wh
    _tnt_count = len(_get_tnt())
    _wh_count  = len(_get_wh())
    st.caption(f"📅 Jan 2023 – Dec 2024  |  {len(df_main)} rows")
    st.caption(f"🏭 {_wh_count} warehouses  |  {_tnt_count} tenants")

# ── KPI helper ────────────────────────────────────────────────────────────────
def kpi_card(label, value, footer="", tag="", tag_class="tag-blue"):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-footer">{footer}</div>
        {'<span class="kpi-tag '+tag_class+'">'+tag+'</span>' if tag else ''}
    </div>"""


def render_kpis(df_kpi):
    kpis = compute_kpis(df_kpi)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(kpi_card(
            "Total Invoiced", inr_fmt(kpis["total_inv"]),
            "incl. 18% GST", "Info", "tag-blue"
        ), unsafe_allow_html=True)
    with c2:
        tc = "tag-green" if kpis["col_rate"] >= 85 else "tag-amber"
        st.markdown(kpi_card(
            "Revenue Collected", inr_fmt(kpis["total_col"]),
            f"{kpis['col_rate']}% collection rate",
            "Healthy" if kpis["col_rate"] >= 85 else "Watch", tc
        ), unsafe_allow_html=True)
    with c3:
        oc = "tag-red" if kpis["total_out"] > 100000 else "tag-amber"
        st.markdown(kpi_card(
            "Outstanding", inr_fmt(kpis["total_out"]),
            f"{kpis['high_risk']} high-risk records",
            "Alert" if kpis["total_out"] > 100000 else "Low", oc
        ), unsafe_allow_html=True)
    with c4:
        pc = "tag-green" if kpis["paid_rate"] >= 85 else "tag-amber"
        st.markdown(kpi_card(
            "Payment Rate", f"{kpis['paid_rate']}%",
            "of all transactions",
            "Healthy" if kpis["paid_rate"] >= 85 else "Review", pc
        ), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card(
            "Avg Monthly Rent", inr_fmt(kpis["avg_rent"]),
            "active contracts", "Info", "tag-blue"
        ), unsafe_allow_html=True)
    with c6:
        dc = "tag-red" if kpis["avg_delay"] > 20 else "tag-amber"
        st.markdown(kpi_card(
            "Avg Delay (when late)", f"{kpis['avg_delay']}d",
            "delayed records only",
            "High" if kpis["avg_delay"] > 20 else "Moderate", dc
        ), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="section-header">📊 Portfolio Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Family warehouse rental business — descriptive summary of all operations</div>', unsafe_allow_html=True)

    kpis = compute_kpis(df)
    # Alerts
    if kpis["total_out"] > 0:
        overdue_tenants = df[df["Balance_Due_INR"] > 0]["Tenant_Name"].unique()
        st.markdown(
            f'<div class="alert-danger">⚠️ <strong>Outstanding Balance Alert:</strong> '
            f'{inr_fmt(kpis["total_out"])} outstanding across {len(overdue_tenants)} tenant(s): '
            f'{", ".join(overdue_tenants[:3])}{"..." if len(overdue_tenants) > 3 else ""}. '
            f'Reminders auto-sent after 10th of month.</div>',
            unsafe_allow_html=True,
        )

    render_kpis(df)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Row 1
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.plotly_chart(monthly_revenue_trend(df), use_container_width=True)
    with c2:
        st.plotly_chart(payment_status_donut(df), use_container_width=True)
    with c3:
        st.plotly_chart(wh_type_revenue(df), use_container_width=True)

    # Row 2
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(location_bar(df), use_container_width=True)
    with c2:
        st.plotly_chart(payment_behaviour_bar(df), use_container_width=True)

    # Segment summary
    st.markdown("#### Tenant Segment Summary")
    seg_counts = df["Tenant_Segment"].value_counts()
    total_seg  = len(df)
    seg_colours = {
        "High-Value Regular": ("#EBF2FA", "#1A3C5E"),
        "Regular Payer":      ("#E8F6F0", "#1D8A5F"),
        "Late Payer":         ("#FDF6E3", "#D97706"),
        "Defaulter":          ("#FCEAEA", "#C0392B"),
    }
    cols = st.columns(4)
    for i, (seg, cnt) in enumerate(seg_counts.items()):
        bg, fg = seg_colours.get(seg, ("#F7F4EF", "#0F1923"))
        pct = round(cnt / total_seg * 100, 1)
        with cols[i % 4]:
            st.markdown(
                f'<div style="background:{bg};border-radius:12px;padding:16px;text-align:center;">'
                f'<div style="font-size:11px;font-weight:600;color:{fg};text-transform:uppercase;'
                f'letter-spacing:.06em;margin-bottom:6px">{seg}</div>'
                f'<div style="font-size:32px;font-weight:700;color:{fg}">{cnt}</div>'
                f'<div style="font-size:11px;color:{fg};opacity:.7">{pct}% of transactions</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(quarterly_revenue(df), use_container_width=True)
    with c2:
        st.plotly_chart(automation_summary(df), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: REVENUE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "💰 Revenue Analysis":
    st.markdown('<div class="section-header">💰 Revenue Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Owner-wise, location-wise and quarterly revenue breakdown</div>', unsafe_allow_html=True)

    render_kpis(df)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(owner_revenue_bar(df), use_container_width=True)
    with c2:
        st.plotly_chart(owner_revenue_pie(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(quarterly_revenue(df), use_container_width=True)
    with c2:
        st.plotly_chart(location_type_heatmap(df), use_container_width=True)

    # Location performance table
    st.markdown("#### Location-wise Performance Table")
    loc_tbl = (
        df.groupby("Warehouse_Location")
        .agg(
            Revenue=("Revenue_Collected_INR", "sum"),
            Invoiced=("Total_Invoice_INR", "sum"),
            Outstanding=("Balance_Due_INR", "sum"),
            Avg_Rent=("Monthly_Rent_INR", "mean"),
            Transactions=("Row_ID", "count"),
            Paid_Rate=("Is_Paid_Binary", "mean"),
        )
        .reset_index()
    )
    loc_tbl["Collection_Rate"] = (loc_tbl["Revenue"] / loc_tbl["Invoiced"] * 100).round(1)
    loc_tbl["Paid_Rate"]        = (loc_tbl["Paid_Rate"] * 100).round(1)
    loc_tbl["Revenue"]          = loc_tbl["Revenue"].apply(inr_fmt)
    loc_tbl["Invoiced"]         = loc_tbl["Invoiced"].apply(inr_fmt)
    loc_tbl["Outstanding"]      = loc_tbl["Outstanding"].apply(inr_fmt)
    loc_tbl["Avg_Rent"]         = loc_tbl["Avg_Rent"].apply(lambda v: inr_fmt(round(v)))
    loc_tbl.columns = ["Location", "Revenue", "Invoiced", "Outstanding",
                       "Avg Rent", "Transactions", "Paid Rate %", "Collection Rate %"]
    st.dataframe(loc_tbl, use_container_width=True, hide_index=True)

    # Owner table
    st.markdown("#### Owner-wise Revenue Table")
    own_tbl = (
        df.groupby(["Owner_ID", "Owner_Name"])
        .agg(
            Revenue=("Revenue_Collected_INR", "sum"),
            Invoiced=("Total_Invoice_INR", "sum"),
            Outstanding=("Balance_Due_INR", "sum"),
            Warehouses=("WH_ID", "nunique"),
            Tenants=("Tenant_ID", "nunique"),
            Paid_Rate=("Is_Paid_Binary", "mean"),
        )
        .reset_index()
    )
    own_tbl["Paid_Rate"] = (own_tbl["Paid_Rate"] * 100).round(1)
    own_tbl["Revenue"]   = own_tbl["Revenue"].apply(inr_fmt)
    own_tbl["Invoiced"]  = own_tbl["Invoiced"].apply(inr_fmt)
    own_tbl["Outstanding"] = own_tbl["Outstanding"].apply(inr_fmt)
    own_tbl.columns = ["Owner ID", "Owner Name", "Revenue Collected",
                       "Total Invoiced", "Outstanding", "Warehouses", "Tenants", "Paid Rate %"]
    st.dataframe(own_tbl, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DIAGNOSTIC ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Diagnostic Analysis":
    st.markdown('<div class="section-header">🔍 Diagnostic Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Why did it happen? — Correlation analysis, delay drivers, and risk patterns</div>', unsafe_allow_html=True)

    st.markdown("#### Correlation Heatmap — All Key Variables")
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    st.markdown(
        '<div class="alert-info">📌 <strong>Key Correlations:</strong> WH Type vs Rent (strong positive), '
        'Risk Score vs Delay Days (strong positive), Customer Tenure vs Delay Days (negative — '
        'longer-tenured tenants are more reliable payers).</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(rent_vs_size(df), use_container_width=True)
    with c2:
        st.plotly_chart(rent_vs_type(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(delay_by_industry(df), use_container_width=True)
    with c2:
        st.plotly_chart(delay_by_tenant_type(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(risk_score_histogram(df), use_container_width=True)
    with c2:
        st.plotly_chart(location_type_heatmap(df), use_container_width=True)

    # Pearson r table
    st.markdown("#### Pearson Correlation Coefficients")
    from scipy.stats import pearsonr
    pairs = [
        ("Monthly_Rent_INR", "Size_Encoded",           "Rent vs WH Size"),
        ("Monthly_Rent_INR", "Type_Encoded",            "Rent vs WH Type"),
        ("Delay_Days",       "TenantType_Encoded",      "Delay vs Tenant Type"),
        ("Delay_Days",       "Customer_Tenure_Months",  "Delay vs Tenure"),
        ("Risk_Score",       "Delay_Days",               "Risk Score vs Delay"),
        ("Is_Paid_Binary",   "Customer_Tenure_Months",  "Paid vs Tenure"),
        ("Monthly_Rent_INR", "Lease_Duration_Months",   "Rent vs Lease Duration"),
    ]
    corr_rows = []
    for c1n, c2n, label in pairs:
        if c1n in df.columns and c2n in df.columns:
            sub = df[[c1n, c2n]].dropna()
            if len(sub) > 5:
                r, p = pearsonr(sub[c1n], sub[c2n])
                strength = (
                    "Strong" if abs(r) > 0.7
                    else "Moderate" if abs(r) > 0.4
                    else "Weak"
                )
                direction = "Positive ↑" if r > 0 else "Negative ↓"
                corr_rows.append({
                    "Variables": label,
                    "Pearson r": round(r, 3),
                    "p-value":   round(p, 4),
                    "Strength":  strength,
                    "Direction": direction,
                    "Significant": "Yes ✓" if p < 0.05 else "No",
                })
    corr_df = pd.DataFrame(corr_rows)
    st.dataframe(corr_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictive — Classification":
    st.markdown('<div class="section-header">🤖 Predictive — Payment Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Random Forest: Predicting whether a tenant will pay or default</div>', unsafe_allow_html=True)

    with st.spinner("Training Random Forest classifier..."):
        model_clf, scaler_clf, metrics_clf, feat_clf = train_classifier(df_main)

    # Performance metrics
    st.markdown("#### Model Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    metric_data = [
        (m1, "Accuracy",  f"{metrics_clf['accuracy']*100:.1f}%",  "tag-green" if metrics_clf['accuracy'] > 0.7 else "tag-amber"),
        (m2, "Precision", f"{metrics_clf['precision']*100:.1f}%", "tag-green" if metrics_clf['precision'] > 0.7 else "tag-amber"),
        (m3, "Recall",    f"{metrics_clf['recall']*100:.1f}%",    "tag-green" if metrics_clf['recall'] > 0.7 else "tag-amber"),
        (m4, "F1-Score",  f"{metrics_clf['f1']*100:.1f}%",        "tag-green" if metrics_clf['f1'] > 0.7 else "tag-amber"),
        (m5, "ROC-AUC",   f"{metrics_clf['roc_auc']:.3f}",        "tag-green" if metrics_clf['roc_auc'] > 0.75 else "tag-amber"),
    ]
    for col, lbl, val, tc in metric_data:
        with col:
            st.markdown(kpi_card(lbl, val, "", lbl, tc), unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            roc_curve_chart(metrics_clf["fpr"], metrics_clf["tpr"], metrics_clf["roc_auc"]),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            confusion_matrix_chart(metrics_clf["cm"]),
            use_container_width=True,
        )

    st.plotly_chart(
        feature_importance_chart(metrics_clf["feature_names"], metrics_clf["feature_importance"]),
        use_container_width=True,
    )

    # Detailed classification report
    st.markdown("#### Detailed Classification Report")
    report = metrics_clf["report"]
    rep_rows = []
    for cls, vals in report.items():
        if isinstance(vals, dict):
            lbl = "Not Paid" if cls == "0" else "Paid" if cls == "1" else cls.title()
            rep_rows.append({
                "Class":     lbl,
                "Precision": round(vals["precision"], 3),
                "Recall":    round(vals["recall"], 3),
                "F1-Score":  round(vals["f1-score"], 3),
                "Support":   int(vals["support"]),
            })
    st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="alert-success">✅ <strong>Business Insight:</strong> '
        'Customer_Tenure_Months and Monthly_Rent_INR are the strongest predictors of payment. '
        'Use this model to screen new tenants — score them before signing the lease.</div>',
        unsafe_allow_html=True,
    )

    # Manual prediction tool
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("#### 🔮 Single Tenant Prediction")
    with st.form("clf_form"):
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            p_rent   = st.number_input("Monthly Rent (₹)", 5000, 500000, 45000, step=1000)
            p_size   = st.selectbox("WH Size", ["Small", "Medium", "Large"])
        with cc2:
            p_type   = st.selectbox("WH Type", ["General", "Dry Warehouse", "Cold Storage", "Distribution Hub"])
            p_ttype  = st.selectbox("Tenant Type", ["Business", "Individual"])
        with cc3:
            p_tenure = st.slider("Tenure (months)", 1, 60, 12)
            p_lease  = st.slider("Lease Duration (months)", 6, 36, 12)
        with cc4:
            p_risk   = st.slider("Est. Risk Score", 0, 100, 30)
            p_mnum   = st.slider("Month Number", 1, 30, 24)
        submitted = st.form_submit_button("Predict Payment Likelihood", type="primary")

    if submitted:
        from utils import encode_prospect, SIZE_MAP, TYPE_MAP, TTYPE_MAP
        enc = encode_prospect({
            "Monthly_Rent_INR": p_rent, "Warehouse_Size": p_size,
            "Warehouse_Type": p_type, "Tenant_Type": p_ttype,
            "Customer_Tenure_Months": p_tenure, "Lease_Duration_Months": p_lease,
            "Risk_Score": p_risk,
        }, month_num=p_mnum)
        row = pd.DataFrame([enc])[feat_clf]
        row_s = scaler_clf.transform(row)
        prob  = model_clf.predict_proba(row_s)[0][1]
        pred  = model_clf.predict(row_s)[0]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(prospect_gauge(prob, "Tenant"), use_container_width=True)
        with c2:
            if prob >= 0.75:
                st.markdown('<div class="alert-success">✅ <strong>Low Risk</strong> — Standard contract recommended.</div>', unsafe_allow_html=True)
            elif prob >= 0.5:
                st.markdown('<div class="alert-info">⚠️ <strong>Medium Risk</strong> — Increase security deposit to 2 months.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-danger">🚫 <strong>High Risk</strong> — Require 3-month advance or decline tenancy.</div>', unsafe_allow_html=True)
            st.metric("Pay Probability", f"{prob*100:.1f}%")
            st.metric("Prediction", "Will Pay ✓" if pred == 1 else "At Risk ✗")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: REGRESSION
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Predictive — Regression":
    st.markdown('<div class="section-header">📈 Predictive — Regression Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Predicting Monthly Rent and Delay Days using Ridge Regression</div>', unsafe_allow_html=True)

    with st.spinner("Training regression models..."):
        model_rent, metrics_rent, feat_rent     = train_rent_regressor(df_main)
        model_delay, metrics_delay, feat_delay  = train_delay_regressor(df_main)

    # Rent model
    st.markdown("#### Model 1 — Monthly Rent Prediction")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(kpi_card("R² Score", f"{metrics_rent['r2']:.3f}", "Variance explained", "Good" if metrics_rent['r2'] > 0.6 else "Fair", "tag-green" if metrics_rent['r2'] > 0.6 else "tag-amber"), unsafe_allow_html=True)
    with r2:
        st.markdown(kpi_card("MAE", inr_fmt(metrics_rent['mae']), "Mean Absolute Error", "Info", "tag-blue"), unsafe_allow_html=True)
    with r3:
        st.markdown(kpi_card("RMSE", inr_fmt(metrics_rent['rmse']), "Root Mean Sq Error", "Info", "tag-blue"), unsafe_allow_html=True)

    st.plotly_chart(
        regression_actual_vs_predicted(
            metrics_rent["y_test"], metrics_rent["y_pred"],
            f"Monthly Rent — Actual vs Predicted (R²={metrics_rent['r2']:.3f})"
        ),
        use_container_width=True,
    )

    # Coefficients
    coef_df = pd.DataFrame({
        "Feature":     feat_rent,
        "Coefficient": metrics_rent["coef"].round(2),
        "Abs Impact":  np.abs(metrics_rent["coef"]).round(2),
    }).sort_values("Abs Impact", ascending=False)
    st.markdown("**Regression Coefficients — Rent Model**")
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Delay model
    st.markdown("#### Model 2 — Delay Days Prediction")
    if model_delay and metrics_delay:
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(kpi_card("R² Score", f"{metrics_delay['r2']:.3f}", "Variance explained", "Info", "tag-blue"), unsafe_allow_html=True)
        with d2:
            st.markdown(kpi_card("MAE", f"{metrics_delay['mae']:.1f} days", "Mean Absolute Error", "Info", "tag-blue"), unsafe_allow_html=True)
        with d3:
            st.markdown(kpi_card("RMSE", f"{metrics_delay['rmse']:.1f} days", "Root Mean Sq Error", "Info", "tag-blue"), unsafe_allow_html=True)

        st.plotly_chart(
            regression_actual_vs_predicted(
                metrics_delay["y_test"], metrics_delay["y_pred"],
                f"Delay Days — Actual vs Predicted (R²={metrics_delay['r2']:.3f})"
            ),
            use_container_width=True,
        )
        st.markdown(
            '<div class="alert-info">📌 <strong>Insight:</strong> Delay days are harder to predict from '
            'structural features alone — delay is behavioural. The classification model (Paid/Not Paid) '
            'is a more reliable operational predictor than delay regression.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Insufficient delayed records to train delay regression model with current filters.")

    # Rent predictor tool
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("#### 🔮 Rent Estimator — What should I charge?")
    with st.form("rent_form"):
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            r_size  = st.selectbox("WH Size",  ["Small", "Medium", "Large"], key="rsize")
        with rc2:
            r_type  = st.selectbox("WH Type",  ["General", "Dry Warehouse", "Cold Storage", "Distribution Hub"], key="rtype")
        with rc3:
            r_lease = st.slider("Lease Duration (months)", 6, 36, 12, key="rlease")
        r_tenure = st.slider("Tenant Tenure (months)", 1, 60, 12, key="rtenure")
        r_mnum   = st.slider("Month Number (1=Jan23)", 1, 30, 25, key="rmnum")
        r_sub = st.form_submit_button("Estimate Fair Rent", type="primary")
    if r_sub:
        from utils import SIZE_MAP, TYPE_MAP
        X_new = pd.DataFrame([{
            "Size_Encoded":           SIZE_MAP.get(r_size, 2),
            "Type_Encoded":           TYPE_MAP.get(r_type, 2),
            "Month_Num":              r_mnum,
            "Customer_Tenure_Months": r_tenure,
            "Lease_Duration_Months":  r_lease,
        }])[feat_rent]
        pred_rent = model_rent.predict(X_new)[0]
        st.success(f"💰 Estimated Fair Rent: **{inr_fmt(max(5000, pred_rent))}** / month")
        st.caption("Based on warehouse size, type, and market conditions from training data.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Clustering":
    st.markdown('<div class="section-header">🎯 K-Means Tenant Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Segment tenants into strategic groups for targeted management</div>', unsafe_allow_html=True)

    k_val = st.slider("Select Number of Clusters (K)", 2, 8, 4, key="k_slider")

    with st.spinner("Running K-Means clustering..."):
        km_model, km_scaler, df_clustered, profile, inertias, feat_cluster = train_clustering(df_main, k=k_val)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(elbow_chart(inertias), use_container_width=True)
    with c2:
        st.plotly_chart(cluster_profile_radar(profile), use_container_width=True)

    st.plotly_chart(cluster_scatter(df_clustered), use_container_width=True)

    # Cluster profile table
    st.markdown("#### Cluster Profiles")
    profile_display = profile.copy()
    profile_display["Avg_Rent"]  = profile_display["Avg_Rent"].apply(lambda v: inr_fmt(round(v)))
    profile_display["Paid_Rate"] = (profile_display["Paid_Rate"] * 100).round(1).astype(str) + "%"
    profile_display["Avg_Delay"] = profile_display["Avg_Delay"].round(1).astype(str) + " days"
    profile_display.index.name = "Cluster ID"
    st.dataframe(profile_display.reset_index(), use_container_width=True, hide_index=True)

    # Segment transactions
    st.markdown("#### Transactions by Cluster")
    cluster_col = "Cluster_Label" if "Cluster_Label" in df_clustered.columns else "KMeans_Cluster"
    show_cols = [c for c in ["Tenant_Name", "Warehouse_Location", "Warehouse_Type",
                              "Monthly_Rent_INR", "Delay_Days", "Payment_Behavior",
                              "Risk_Score", cluster_col] if c in df_clustered.columns]
    selected_cluster = st.selectbox("Filter by Cluster", ["All"] + sorted(df_clustered[cluster_col].unique().tolist()))
    view_df = df_clustered if selected_cluster == "All" else df_clustered[df_clustered[cluster_col] == selected_cluster]
    st.dataframe(view_df[show_cols].head(50), use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="alert-success">✅ <strong>Prescriptive Actions by Segment:</strong><br>'
        '• <strong>High-Value Regular</strong>: Offer 3-year lock-in with minor rent relief<br>'
        '• <strong>Regular Payer</strong>: Annual rent revision at CPI + 2%<br>'
        '• <strong>Late Payer</strong>: Early payment discount (1% if paid by 5th); penalty after 10th<br>'
        '• <strong>At-Risk</strong>: Legal notice + non-renewal notice; require 3-month advance</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":
    st.markdown('<div class="section-header">🔗 Association Rules Mining</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Discovering patterns: Tenant Type + WH Type → Payment Behaviour</div>', unsafe_allow_html=True)

    min_sup = st.slider("Minimum Support", 0.01, 0.20, 0.05, 0.01, key="assoc_sup")

    rules = compute_association_rules(df_main, min_support=min_sup)

    if rules.empty:
        st.warning("No rules found at this support threshold. Lower the minimum support.")
    else:
        # Summary metrics
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.markdown(kpi_card("Total Rules", str(len(rules)), "at current threshold", "Info", "tag-blue"), unsafe_allow_html=True)
        with a2:
            best_lift = rules["Lift"].max()
            st.markdown(kpi_card("Max Lift", f"{best_lift:.2f}", "strongest rule", "High" if best_lift > 1.2 else "Moderate", "tag-green" if best_lift > 1.2 else "tag-amber"), unsafe_allow_html=True)
        with a3:
            best_conf = rules["Confidence"].max()
            st.markdown(kpi_card("Max Confidence", f"{best_conf*100:.1f}%", "most reliable rule", "High", "tag-green"), unsafe_allow_html=True)
        with a4:
            high_lift = (rules["Lift"] > 1.1).sum()
            st.markdown(kpi_card("High-Lift Rules", str(high_lift), "Lift > 1.1", "Actionable", "tag-orange" if high_lift > 0 else "tag-blue"), unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(association_heatmap(rules), use_container_width=True)
        with c2:
            st.plotly_chart(association_bubble(rules), use_container_width=True)

        # Full rules table
        st.markdown("#### All Association Rules — Sorted by Lift ↓")
        display_rules = rules[["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Count"]].copy()
        display_rules["Support"]    = (display_rules["Support"] * 100).round(2).astype(str) + "%"
        display_rules["Confidence"] = (display_rules["Confidence"] * 100).round(1).astype(str) + "%"
        display_rules["Lift"]       = display_rules["Lift"].round(3)
        st.dataframe(display_rules, use_container_width=True, hide_index=True)

        # Interpretation
        top_rule = rules.iloc[0]
        st.markdown(
            f'<div class="alert-success">🏆 <strong>Strongest Rule:</strong> '
            f'"{top_rule["Antecedent"]}" → "{top_rule["Consequent"]}" | '
            f'Support: {top_rule["Support"]*100:.1f}% | '
            f'Confidence: {top_rule["Confidence"]*100:.1f}% | '
            f'Lift: {top_rule["Lift"]:.2f} — '
            f'This combination is {top_rule["Lift"]:.1f}× more likely to result in '
            f'"{top_rule["Consequent"]}" than random chance.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="alert-info">📌 <strong>Lift Interpretation:</strong> '
            'Lift > 1 = positive association (better than chance) | '
            'Lift = 1 = no association | Lift < 1 = negative association. '
            'Use high-lift rules to decide which tenant type to target for each warehouse type.</div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PROSPECT SCORING
# ════════════════════════════════════════════════════════════════════════════
elif page == "🚀 Prospect Scoring":
    st.markdown('<div class="section-header">🚀 Prospect Scoring & Lead Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload new prospect data and predict payment likelihood to prioritise your marketing strategy</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="alert-info">💡 <strong>How it works:</strong> Upload a CSV with prospect details. '
        'The trained Random Forest model scores each prospect on payment probability (0–100%), '
        'assigns a risk tier, and recommends a contract approach.</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading classification model..."):
        model_clf, scaler_clf, metrics_clf, feat_clf = train_classifier(df_main)

    # Template download
    template_df = pd.DataFrame([
        {"Tenant_Name": "ABC Logistics",   "Tenant_Type": "Business",   "Industry_Type": "Logistics",
         "Warehouse_Type": "Distribution Hub", "Warehouse_Size": "Large",  "Monthly_Rent_INR": 90000,
         "Customer_Tenure_Months": 0, "Lease_Duration_Months": 24},
        {"Tenant_Name": "Priya Textiles",  "Tenant_Type": "Business",   "Industry_Type": "Textile",
         "Warehouse_Type": "Dry Warehouse",   "Warehouse_Size": "Medium", "Monthly_Rent_INR": 35000,
         "Customer_Tenure_Months": 0, "Lease_Duration_Months": 12},
        {"Tenant_Name": "Ravi Kumar",      "Tenant_Type": "Individual",  "Industry_Type": "Trading",
         "Warehouse_Type": "General",         "Warehouse_Size": "Small",  "Monthly_Rent_INR": 18000,
         "Customer_Tenure_Months": 0, "Lease_Duration_Months": 6},
    ])

    buf = io.BytesIO()
    template_df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(
        "⬇️ Download Prospect Template CSV",
        data=buf,
        file_name="prospect_template.csv",
        mime="text/csv",
    )

    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload Prospect CSV",
        type=["csv"],
        help="Use the template above. Required columns: Tenant_Name, Tenant_Type, Industry_Type, "
             "Warehouse_Type, Warehouse_Size, Monthly_Rent_INR, Customer_Tenure_Months, Lease_Duration_Months",
    )

    # Manual entry fallback
    use_manual = st.checkbox("Or enter prospect details manually")

    prospects_df = None

    if uploaded is not None:
        try:
            prospects_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(prospects_df)} prospects from file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    elif use_manual:
        with st.form("manual_prospect"):
            st.markdown("**Enter Prospect Details**")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                m_name   = st.text_input("Tenant Name", "New Prospect Ltd")
                m_ttype  = st.selectbox("Tenant Type",  ["Business", "Individual"])
                m_ind    = st.text_input("Industry",     "Logistics")
            with mc2:
                m_wtype  = st.selectbox("WH Type",  ["Distribution Hub", "Cold Storage", "Dry Warehouse", "General"])
                m_wsize  = st.selectbox("WH Size",  ["Large", "Medium", "Small"])
                m_rent   = st.number_input("Monthly Rent (₹)", 5000, 300000, 50000, step=1000)
            with mc3:
                m_tenure = st.slider("Est. Tenure (months)", 0, 60, 0)
                m_lease  = st.slider("Lease Duration (months)", 6, 36, 12)
            m_sub = st.form_submit_button("Score This Prospect", type="primary")
        if m_sub:
            prospects_df = pd.DataFrame([{
                "Tenant_Name": m_name, "Tenant_Type": m_ttype, "Industry_Type": m_ind,
                "Warehouse_Type": m_wtype, "Warehouse_Size": m_wsize, "Monthly_Rent_INR": m_rent,
                "Customer_Tenure_Months": m_tenure, "Lease_Duration_Months": m_lease,
            }])

    if prospects_df is not None and not prospects_df.empty:
        try:
            scored = score_prospects(prospects_df, model_clf, scaler_clf, feat_clf)
            st.markdown("### 🎯 Scoring Results")

            # Gauges for up to 3
            if len(scored) <= 3:
                gcols = st.columns(len(scored))
                for i, (_, row) in enumerate(scored.iterrows()):
                    with gcols[i]:
                        name = row.get("Tenant_Name", f"Prospect {i+1}")
                        st.plotly_chart(prospect_gauge(float(row["Pay_Probability"]), name), use_container_width=True)
            else:
                st.plotly_chart(prospect_risk_bar(scored), use_container_width=True)

            # Results table
            show_cols = ["Tenant_Name", "Tenant_Type", "Industry_Type",
                         "Warehouse_Type", "Warehouse_Size", "Monthly_Rent_INR",
                         "Pay_Probability", "Predicted_Payment", "Risk_Tier", "Recommended_Action"]
            out_df = scored[[c for c in show_cols if c in scored.columns]].copy()
            out_df["Pay_Probability"] = (out_df["Pay_Probability"] * 100).round(1).astype(str) + "%"
            st.dataframe(out_df, use_container_width=True, hide_index=True)

            # Download scored results
            scored_buf = io.BytesIO()
            scored.to_csv(scored_buf, index=False)
            scored_buf.seek(0)
            st.download_button(
                "⬇️ Download Scored Results CSV",
                data=scored_buf,
                file_name="prospect_scores.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Scoring error: {e}")
            st.info("Make sure your CSV has all required columns. Download the template above.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: INVOICE & REMINDERS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📨 Invoice & Reminders":
    render_notification_page(df)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Data Explorer":
    st.markdown('<div class="section-header">📋 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Browse, filter, and download the full dataset</div>', unsafe_allow_html=True)

    render_kpis(df)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Column selector
    all_cols = df.columns.tolist()
    default_cols = [c for c in [
        "Row_ID", "Tenant_Name", "Owner_Name", "Warehouse_Location",
        "Warehouse_Type", "Warehouse_Size", "Month_Name", "Monthly_Rent_INR",
        "Total_Invoice_INR", "Payment_Status", "Payment_Behavior",
        "Delay_Days", "Balance_Due_INR", "Risk_Score", "Tenant_Segment",
    ] if c in all_cols]
    selected_cols = st.multiselect("Select columns to display", all_cols, default=default_cols)
    if not selected_cols:
        selected_cols = default_cols

    search = st.text_input("Search by tenant name", "")
    view   = df.copy()
    if search:
        mask = view.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
        view = view[mask]

    st.dataframe(view[selected_cols], use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(view)} rows × {len(selected_cols)} columns")

    # Download
    dl_buf = io.BytesIO()
    view[selected_cols].to_csv(dl_buf, index=False)
    dl_buf.seek(0)
    st.download_button(
        "⬇️ Download Filtered Dataset (CSV)",
        data=dl_buf,
        file_name="warehouse_data_filtered.csv",
        mime="text/csv",
    )

    # Descriptive stats
    st.markdown("#### Descriptive Statistics")
    num_cols = view.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.dataframe(view[num_cols].describe().round(2), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    render_settings_page()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ADMIN
# ════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Admin — Manage Data":
    render_admin_page()
