import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Try XGBoost first, fallback if unavailable
MODEL_NAME = "XGBoost"
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False
    MODEL_NAME = "Gradient Boosting"


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Aurum Retail Intelligence",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# THEME / CSS
# =========================
st.markdown(
    """
    <style>
        :root {
            --bg: #0b0b0d;
            --panel: #121214;
            --panel-2: #17171a;
            --gold: #d4af37;
            --gold-soft: #8f7530;
            --text: #f3f3f3;
            --muted: #a8a8a8;
            --border: rgba(212, 175, 55, 0.22);
            --success: #1f9d55;
            --danger: #d64545;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(212,175,55,0.08), transparent 22%),
                linear-gradient(135deg, #090909 0%, #0d0d10 45%, #0a0a0c 100%);
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b0b0d 0%, #111113 100%);
            border-right: 1px solid var(--border);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(18,18,20,0.95), rgba(13,13,15,0.92));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 28px 30px;
            box-shadow: 0 0 0 1px rgba(212,175,55,0.04), 0 18px 40px rgba(0,0,0,0.35);
        }

        .metric-card {
            background: linear-gradient(180deg, rgba(20,20,22,0.98), rgba(14,14,16,0.98));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 16px 18px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.25);
        }

        .section-title {
            color: var(--gold);
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
            letter-spacing: 0.3px;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .gold-label {
            color: var(--gold);
            font-weight: 700;
            letter-spacing: 0.3px;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(20,20,22,0.98), rgba(14,14,16,0.98));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 10px 12px;
        }

        div[data-testid="stMetricLabel"] {
            color: #cfcfcf;
        }

        div[data-testid="stMetricValue"] {
            color: white;
        }

        .insight-box {
            background: rgba(18,18,20,0.96);
            border: 1px solid var(--border);
            border-left: 4px solid var(--gold);
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(20,20,22,0.9);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 14px;
            color: white;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(212,175,55,0.16), rgba(212,175,55,0.07));
        }

        .stButton>button {
            background: linear-gradient(180deg, #d4af37, #b8942f);
            color: black;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.6rem 1rem;
        }

        .stDownloadButton>button {
            background: linear-gradient(180deg, #d4af37, #b8942f);
            color: black;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.6rem 1rem;
        }

        .stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label, .stTextInput label {
            color: #efefef !important;
            font-weight: 600;
        }

        hr {
            border-color: rgba(212,175,55,0.18);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
def gold_template(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f3f3"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(212,175,55,0.15)",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def format_money(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"


def add_hero(title: str, subtitle: str, right_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div style="display:flex; justify-content:space-between; gap:18px; align-items:flex-start; flex-wrap:wrap;">
                <div style="max-width: 760px;">
                    <div class="gold-label" style="font-size:0.82rem;">AURUM RETAIL INTELLIGENCE</div>
                    <div style="font-size:2.35rem; line-height:1.15; font-weight:700; margin-top:8px;">{title}</div>
                    <div class="small-muted" style="margin-top:10px; max-width:700px;">{subtitle}</div>
                </div>
                <div style="min-width:240px; flex: 0 0 280px;" class="metric-card">
                    <div class="small-muted">{right_text}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# DATA LOADING
# =========================
import re

@st.cache_data(show_spinner=False)
def load_raw_data(file_path: str) -> pd.DataFrame:
    file_path = file_path.strip()

    # لو لينك Google Drive
    if "drive.google.com" in file_path:
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", file_path)
        if not match:
            raise ValueError("Invalid Google Drive link")

        file_id = match.group(1)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        try:
            df = pd.read_excel(download_url)
        except:
            df = pd.read_csv(download_url)

    # لو path عادي
    else:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def prepare_datasets(file_path: str):
    df = load_raw_data(file_path).copy()

    # InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Raw sales dataset for broad business analysis
    df_sales = df.copy()
    df_sales = df_sales.dropna(subset=["InvoiceDate"])
    df_sales["Description"] = df_sales["Description"].fillna("Unknown Product")
    df_sales["TotalPrice"] = df_sales["Quantity"] * df_sales["UnitPrice"]
    df_sales["Year"] = df_sales["InvoiceDate"].dt.year
    df_sales["Month"] = df_sales["InvoiceDate"].dt.month
    df_sales["MonthName"] = df_sales["InvoiceDate"].dt.strftime("%b")
    df_sales["Day"] = df_sales["InvoiceDate"].dt.day
    df_sales["Hour"] = df_sales["InvoiceDate"].dt.hour
    df_sales["IsReturn"] = df_sales["Quantity"] <= 0

    # Clean customer dataset for modeling / RFM
    df_clean = df.copy()
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"], errors="coerce")
    df_clean = df_clean.dropna(subset=["InvoiceDate", "CustomerID", "Description"])
    df_clean = df_clean[df_clean["Quantity"] > 0]
    df_clean = df_clean[df_clean["UnitPrice"] > 0]
    df_clean = df_clean.drop_duplicates()
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(str)
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    df_clean["Year"] = df_clean["InvoiceDate"].dt.year
    df_clean["Month"] = df_clean["InvoiceDate"].dt.month
    df_clean["MonthName"] = df_clean["InvoiceDate"].dt.strftime("%b")
    df_clean["Day"] = df_clean["InvoiceDate"].dt.day
    df_clean["Hour"] = df_clean["InvoiceDate"].dt.hour

    return df_sales, df_clean


@st.cache_data(show_spinner=False)
def build_rfm(df_clean: pd.DataFrame) -> pd.DataFrame:
    reference_date = df_clean["InvoiceDate"].max()

    rfm = (
        df_clean.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )

    rfm["Avg_Order_Value"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["Is_Frequent"] = (rfm["Frequency"] > 5).astype(int)
    rfm["Is_High_Spender"] = (rfm["Monetary"] > rfm["Monetary"].median()).astype(int)

    # RFM scores
    rfm["R_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop")
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    rfm["M_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")

    rfm["R_score"] = rfm["R_score"].astype(int)
    rfm["F_score"] = rfm["F_score"].astype(int)
    rfm["M_score"] = rfm["M_score"].astype(int)

    # Rule-based segment
    def segment_customer(row):
        if row["R_score"] >= 4 and row["F_score"] >= 4:
            return "VIP"
        elif row["F_score"] >= 4:
            return "Loyal"
        elif row["R_score"] <= 2:
            return "At Risk"
        return "Regular"

    rfm["Segment"] = rfm.apply(segment_customer, axis=1)

    # KMeans clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_stats = (
        rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .reset_index()
    )

    # Lowest recency + highest frequency/monetary => VIP-like
    vip_cluster = cluster_stats.sort_values(
        by=["Recency", "Frequency", "Monetary"],
        ascending=[True, False, False],
    )["Cluster"].iloc[0]

    low_cluster = cluster_stats.sort_values(
        by=["Recency", "Frequency", "Monetary"],
        ascending=[False, True, True],
    )["Cluster"].iloc[0]

    cluster_map = {}
    for c in cluster_stats["Cluster"]:
        if c == vip_cluster:
            cluster_map[c] = "VIP"
        elif c == low_cluster:
            cluster_map[c] = "At Risk"
        else:
            cluster_map[c] = "Regular"

    rfm["Cluster_Label"] = rfm["Cluster"].map(cluster_map)

    # Churn target
    rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

    return rfm


@st.cache_resource(show_spinner=False)
def train_model(rfm: pd.DataFrame):
    feature_cols = [
        "Frequency",
        "Monetary",
        "Avg_Order_Value",
        "Is_Frequent",
        "Is_High_Spender",
    ]
    X = rfm[feature_cols]
    y = rfm["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    if XGB_AVAILABLE:
        scale_pos_weight = safe_div((y_train == 0).sum(), (y_train == 1).sum())
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )
    else:
        model = GradientBoostingClassifier(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    return model, metrics


def get_feature_importance_df(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        vals = np.zeros(len(feature_cols))

    fi = pd.DataFrame({"Feature": feature_cols, "Importance": vals})
    fi = fi.sort_values("Importance", ascending=False)
    return fi


# =========================
# SIDEBAR
# =========================
st.sidebar.markdown(
    """
    <div style="padding: 6px 2px 16px 2px;">
        <div style="color:#d4af37; font-size:1.2rem; font-weight:800;">AURUM RETAIL</div>
        <div style="color:#a8a8a8; font-size:0.88rem;">Analytics • Segmentation • AI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

import re

default_source = "https://drive.google.com/file/d/13CGT3pqJ-MrULXb2tcXPe3z1JxXxdoah/view?usp=sharing"
file_path = st.sidebar.text_input("Data file path or Google Drive link", value=default_source)

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Sales Intelligence",
        "Product Intelligence",
        "Customer Intelligence",
        "AI Prediction",
        "Insights & Strategy",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Engine:** {MODEL_NAME}")


# =========================
# LOAD MAIN OBJECTS
# =========================
try:
    df_sales, df_clean = prepare_datasets(file_path)
    rfm = build_rfm(df_clean)
    model, model_metrics = train_model(rfm)
    feature_importance_df = get_feature_importance_df(model, model_metrics["feature_cols"])
except Exception as e:
    st.error("Failed to load the dataset. Check the file path and column names.")
    st.exception(e)
    st.stop()


# =========================
# GLOBAL FILTERS
# =========================
with st.sidebar:
    st.markdown("### Filters")
    countries = ["All"] + sorted(df_clean["Country"].dropna().astype(str).unique().tolist())
    selected_country = st.selectbox("Country", countries, index=0)

    available_months = sorted(df_clean["Month"].dropna().unique().tolist())
    month_range = st.slider(
        "Month range",
        min_value=int(min(available_months)),
        max_value=int(max(available_months)),
        value=(int(min(available_months)), int(max(available_months))),
    )

if selected_country == "All":
    filtered_df = df_clean[
        (df_clean["Month"] >= month_range[0]) & (df_clean["Month"] <= month_range[1])
    ].copy()
else:
    filtered_df = df_clean[
        (df_clean["Country"] == selected_country)
        & (df_clean["Month"] >= month_range[0])
        & (df_clean["Month"] <= month_range[1])
    ].copy()

filtered_sales_df = filtered_df.copy()

# =========================
# COMMON KPIS
# =========================
total_revenue = filtered_sales_df["TotalPrice"].sum()
total_orders = filtered_sales_df["InvoiceNo"].nunique()
total_customers = filtered_sales_df["CustomerID"].nunique()
total_products = filtered_sales_df["StockCode"].nunique()
avg_order_value = safe_div(total_revenue, total_orders)
returns_count = int((df_sales["Quantity"] <= 0).sum())
returns_rate = safe_div(returns_count, len(df_sales)) * 100


# =========================
# PAGE: EXECUTIVE OVERVIEW
# =========================
if page == "Executive Overview":
    add_hero(
        title="Retail Intelligence for Revenue, Segments, and Churn Risk",
        subtitle=(
            "A premium analytics workspace for understanding sales performance, customer behavior, "
            "high-value segments, and churn risk using business-ready visual storytelling."
        ),
        right_text=(
            f"<div class='gold-label'>Scope</div>"
            f"<div style='font-size:1.35rem; font-weight:700; margin-top:8px;'>{selected_country}</div>"
            f"<div class='small-muted' style='margin-top:8px;'>Months: {month_range[0]} → {month_range[1]}</div>"
        ),
    )
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", format_money(total_revenue))
    c2.metric("Customers", f"{total_customers:,}")
    c3.metric("Orders", f"{total_orders:,}")
    c4.metric("Avg Order Value", format_money(avg_order_value))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Products", f"{total_products:,}")
    c6.metric("Return Rows", f"{returns_count:,}")
    c7.metric("Return Rate", f"{returns_rate:.2f}%")
    c8.metric("VIP Customers", f"{(rfm['Cluster_Label'] == 'VIP').sum():,}")

    st.write("")
    left, right = st.columns([1.35, 1])

    with left:
        monthly = (
            filtered_sales_df.groupby(["Month", "MonthName"], as_index=False)["TotalPrice"]
            .sum()
            .sort_values("Month")
        )
        fig = px.line(
            monthly,
            x="MonthName",
            y="TotalPrice",
            markers=True,
            title="Monthly Revenue Trend",
        )
        fig.update_traces(line=dict(width=4), marker=dict(size=10))
        fig.update_layout(yaxis_title="Revenue", xaxis_title="Month")
        gold_template(fig, 430)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        seg_counts = rfm["Cluster_Label"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            hole=0.58,
            title="Customer Base Composition",
        )
        gold_template(fig, 430)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        country_sales = (
            filtered_sales_df.groupby("Country", as_index=False)["TotalPrice"]
            .sum()
            .sort_values("TotalPrice", ascending=False)
            .head(10)
        )
        fig = px.bar(
            country_sales,
            x="Country",
            y="TotalPrice",
            title="Top Countries by Revenue",
        )
        gold_template(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        hourly = (
            filtered_sales_df.groupby("Hour", as_index=False)["TotalPrice"]
            .sum()
            .sort_values("Hour")
        )
        fig = px.area(
            hourly,
            x="Hour",
            y="TotalPrice",
            title="Hourly Revenue Pattern",
        )
        gold_template(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Executive Commentary")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            <div class="insight-box">
                <b>Revenue concentration:</b> Revenue in the current selection is <span class="gold-label">{format_money(total_revenue)}</span>.
                The business shows strong concentration around a subset of customers and products, with VIPs representing a small share of customers but outsized value.
            </div>
            <div class="insight-box">
                <b>Seasonality:</b> Revenue patterns show end-of-year strength, supporting the thesis that the business is highly seasonal and should prepare inventory and campaigns before peak months.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        churn_share = rfm["Churn"].mean() * 100
        st.markdown(
            f"""
            <div class="insight-box">
                <b>Churn exposure:</b> Approximately <span class="gold-label">{churn_share:.1f}%</span> of customers are labeled as churn-risk by the business definition used in the model.
            </div>
            <div class="insight-box">
                <b>Operational takeaway:</b> The commercial focus should split between defending VIP customers, converting regulars upward, and reactivating at-risk customers with targeted campaigns.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================
# PAGE: SALES INTELLIGENCE
# =========================
elif page == "Sales Intelligence":
    add_hero(
        title="Sales Performance and Temporal Behavior",
        subtitle=(
            "This view explains when revenue happens, how demand evolves over time, and where business intensity is concentrated."
        ),
        right_text=(
            f"<div class='gold-label'>Transactions</div>"
            f"<div style='font-size:1.35rem; font-weight:700; margin-top:8px;'>{len(filtered_sales_df):,}</div>"
            f"<div class='small-muted' style='margin-top:8px;'>Valid customer transactions after cleaning</div>"
        ),
    )
    st.write("")

    tab1, tab2, tab3 = st.tabs(["Monthly", "Hourly", "Country"])

    with tab1:
        monthly = (
            filtered_sales_df.groupby(["Month", "MonthName"], as_index=False)
            .agg(Revenue=("TotalPrice", "sum"), Orders=("InvoiceNo", "nunique"))
            .sort_values("Month")
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(monthly, x="MonthName", y="Revenue", title="Monthly Revenue")
            gold_template(fig, 420)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(monthly, x="MonthName", y="Orders", markers=True, title="Monthly Orders")
            gold_template(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        hourly = (
            filtered_sales_df.groupby("Hour", as_index=False)
            .agg(Revenue=("TotalPrice", "sum"), Orders=("InvoiceNo", "nunique"))
            .sort_values("Hour")
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(hourly, x="Hour", y="Revenue", markers=True, title="Revenue by Hour")
            gold_template(fig, 420)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(hourly, x="Hour", y="Orders", title="Orders by Hour")
            gold_template(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        country = (
            filtered_sales_df.groupby("Country", as_index=False)
            .agg(Revenue=("TotalPrice", "sum"), Customers=("CustomerID", "nunique"))
            .sort_values("Revenue", ascending=False)
            .head(12)
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(country, x="Country", y="Revenue", title="Top Countries by Revenue")
            gold_template(fig, 420)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(
                country,
                x="Customers",
                y="Revenue",
                size="Revenue",
                hover_name="Country",
                title="Country Revenue vs Customer Count",
            )
            gold_template(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sales Narrative")
    st.markdown(
        """
        <div class="insight-box">
            <b>Seasonal shape:</b> The business demonstrates a clear seasonal profile, with revenue accelerating in stronger months and softening in weaker months.
            This supports campaign timing, inventory planning, and budget allocation decisions.
        </div>
        <div class="insight-box">
            <b>Daytime pattern:</b> Customer purchasing behavior is concentrated in business hours, especially late morning to early afternoon,
            suggesting campaigns and operational readiness should align with those peak windows.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# PAGE: PRODUCT INTELLIGENCE
# =========================
elif page == "Product Intelligence":
    add_hero(
        title="Product Mix, Volume Leaders, and Revenue Drivers",
        subtitle=(
            "This section separates quantity leaders from true revenue leaders, helping distinguish popularity from profitability."
        ),
        right_text=(
            f"<div class='gold-label'>Product Catalog</div>"
            f"<div style='font-size:1.35rem; font-weight:700; margin-top:8px;'>{filtered_sales_df['StockCode'].nunique():,}</div>"
            f"<div class='small-muted' style='margin-top:8px;'>Unique products in the selected scope</div>"
        ),
    )
    st.write("")

    top_qty = (
        filtered_sales_df.groupby("Description", as_index=False)["Quantity"]
        .sum()
        .sort_values("Quantity", ascending=False)
        .head(10)
    )

    top_rev = (
        filtered_sales_df.groupby("Description", as_index=False)["TotalPrice"]
        .sum()
        .sort_values("TotalPrice", ascending=False)
        .head(10)
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            top_qty.sort_values("Quantity"),
            x="Quantity",
            y="Description",
            orientation="h",
            title="Top 10 Products by Quantity",
        )
        gold_template(fig, 450)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            top_rev.sort_values("TotalPrice"),
            x="TotalPrice",
            y="Description",
            orientation="h",
            title="Top 10 Products by Revenue",
        )
        gold_template(fig, 450)
        st.plotly_chart(fig, use_container_width=True)

    top3_rev = top_rev.head(3).copy()
    c1, c2 = st.columns([1, 1.15])

    with c1:
        fig = px.pie(
            top3_rev,
            names="Description",
            values="TotalPrice",
            title="Top 3 Revenue Share",
            hole=0.45,
        )
        gold_template(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        merged = top_qty.merge(top_rev, on="Description", how="outer").fillna(0)
        merged = merged.rename(columns={"TotalPrice": "Revenue"})
        fig = px.scatter(
            merged,
            x="Quantity",
            y="Revenue",
            size="Revenue",
            hover_name="Description",
            title="Quantity vs Revenue for Leading Products",
        )
        gold_template(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Product Narrative")
    st.markdown(
        """
        <div class="insight-box">
            <b>Popularity is not profitability:</b> The products that sell the most units are not always the products generating the most revenue.
            This distinction matters for assortment strategy, pricing, and promotions.
        </div>
        <div class="insight-box">
            <b>Concentration risk:</b> A relatively small number of products drives a large share of revenue,
            which creates both an optimization opportunity and a dependency risk if supply is disrupted.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# PAGE: CUSTOMER INTELLIGENCE
# =========================
elif page == "Customer Intelligence":
    add_hero(
        title="Customer Value, Segmentation, and Cluster Behavior",
        subtitle=(
            "This workspace turns transactions into customer intelligence using RFM scoring, rule-based segments, and clustering."
        ),
        right_text=(
            f"<div class='gold-label'>Customer Base</div>"
            f"<div style='font-size:1.35rem; font-weight:700; margin-top:8px;'>{len(rfm):,}</div>"
            f"<div class='small-muted' style='margin-top:8px;'>Customers with valid identifiers and cleaned history</div>"
        ),
    )
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VIP", f"{(rfm['Segment'] == 'VIP').sum():,}")
    c2.metric("Loyal", f"{(rfm['Segment'] == 'Loyal').sum():,}")
    c3.metric("Regular", f"{(rfm['Segment'] == 'Regular').sum():,}")
    c4.metric("At Risk", f"{(rfm['Segment'] == 'At Risk').sum():,}")

    tabs = st.tabs(["Segments", "Clusters", "Top Customers", "Lookup"])

    with tabs[0]:
        left, right = st.columns(2)

        with left:
            seg = rfm["Segment"].value_counts().reset_index()
            seg.columns = ["Segment", "Count"]
            fig = px.bar(seg, x="Segment", y="Count", title="RFM Segment Distribution")
            gold_template(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            avg_seg = (
                rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
                .mean()
                .reset_index()
            )
            fig = px.scatter(
                avg_seg,
                x="Frequency",
                y="Monetary",
                size="Recency",
                color="Segment",
                hover_name="Segment",
                title="Average Segment Profile",
            )
            gold_template(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        left, right = st.columns(2)

        with left:
            cluster_view = rfm.copy()
            fig = px.scatter(
                cluster_view,
                x="Frequency",
                y="Monetary",
                color="Cluster_Label",
                hover_data=["CustomerID", "Recency", "Segment"],
                title="Customer Clusters",
            )
            gold_template(fig, 450)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            cluster_summary = (
                rfm.groupby("Cluster_Label")[["Recency", "Frequency", "Monetary"]]
                .mean()
                .reset_index()
                .sort_values("Monetary", ascending=False)
            )
            st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

    with tabs[2]:
        top_customers = (
            rfm.sort_values("Monetary", ascending=False)
            .head(15)[["CustomerID", "Recency", "Frequency", "Monetary", "Segment", "Cluster_Label"]]
        )
        left, right = st.columns([1.1, 0.9])

        with left:
            fig = px.bar(
                top_customers.sort_values("Monetary"),
                x="Monetary",
                y="CustomerID",
                orientation="h",
                color="Cluster_Label",
                title="Top Customers by Revenue",
            )
            gold_template(fig, 480)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.dataframe(top_customers, use_container_width=True, hide_index=True)

    with tabs[3]:
        customer_id = st.text_input("Enter Customer ID", value="")
        if customer_id:
            row = rfm[rfm["CustomerID"] == customer_id]
            if row.empty:
                st.warning("Customer not found.")
            else:
                customer = row.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Recency", int(customer["Recency"]))
                c2.metric("Frequency", int(customer["Frequency"]))
                c3.metric("Monetary", format_money(customer["Monetary"]))
                c4.metric("Avg Order Value", format_money(customer["Avg_Order_Value"]))

                st.markdown(
                    f"""
                    <div class="insight-box">
                        <b>Rule-based Segment:</b> {customer['Segment']}<br>
                        <b>Cluster Label:</b> {customer['Cluster_Label']}<br>
                        <b>RFM Scores:</b> R={customer['R_score']} • F={customer['F_score']} • M={customer['M_score']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("### Customer Narrative")
    st.markdown(
        """
        <div class="insight-box">
            <b>Portfolio shape:</b> The customer base follows a classic pattern: a very small elite group creates disproportionate value,
            a large regular base creates stability, and a meaningful at-risk group threatens future revenue.
        </div>
        <div class="insight-box">
            <b>Management implication:</b> VIP customers should be defended with premium treatment,
            regular customers should be converted upward, and at-risk customers should receive reactivation campaigns.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# PAGE: AI PREDICTION
# =========================
elif page == "AI Prediction":
    add_hero(
        title="Churn Prediction Engine",
        subtitle=(
            "This page exposes the churn model as an operational decision layer. "
            "Use manual inputs or customer lookup to estimate churn probability and intervention priority."
        ),
        right_text=(
            f"<div class='gold-label'>Model Quality</div>"
            f"<div style='font-size:1.35rem; font-weight:700; margin-top:8px;'>Recall {model_metrics['recall']:.2f}</div>"
            f"<div class='small-muted' style='margin-top:8px;'>Best metric for catching churn risk</div>"
        ),
    )
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{model_metrics['accuracy']:.2f}")
    c2.metric("Precision", f"{model_metrics['precision']:.2f}")
    c3.metric("Recall", f"{model_metrics['recall']:.2f}")
    c4.metric("F1 Score", f"{model_metrics['f1']:.2f}")

    tab1, tab2, tab3 = st.tabs(["Manual Prediction", "Customer Lookup", "Model Diagnostics"])

    with tab1:
        left, right = st.columns([0.95, 1.05])

        with left:
            frequency = st.number_input("Frequency", min_value=1, value=3, step=1)
            monetary = st.number_input("Monetary", min_value=1.0, value=500.0, step=50.0)
            avg_order_value = st.number_input(
                "Avg Order Value",
                min_value=1.0,
                value=float(max(1.0, monetary / max(frequency, 1))),
                step=10.0,
            )
            is_frequent = st.selectbox("Is Frequent Customer?", [0, 1], index=1 if frequency > 5 else 0)
            is_high_spender = st.selectbox("Is High Spender?", [0, 1], index=0)

            if st.button("Predict Churn"):
                sample = pd.DataFrame(
                    [{
                        "Frequency": frequency,
                        "Monetary": monetary,
                        "Avg_Order_Value": avg_order_value,
                        "Is_Frequent": is_frequent,
                        "Is_High_Spender": is_high_spender,
                    }]
                )
                pred = int(model.predict(sample)[0])
                prob = float(model.predict_proba(sample)[0][1]) if hasattr(model, "predict_proba") else float(pred)

                if pred == 1:
                    st.error(f"High churn risk • Probability: {prob:.2%}")
                else:
                    st.success(f"Likely active • Probability of churn: {prob:.2%}")

        with right:
            st.markdown("#### Interpretation Guide")
            st.markdown(
                """
                <div class="insight-box">
                    <b>High Frequency + High Monetary</b> usually indicates a healthier customer profile.
                </div>
                <div class="insight-box">
                    <b>Low Frequency + Low Monetary</b> tends to push customers toward higher churn risk.
                </div>
                <div class="insight-box">
                    <b>Use case:</b> This model is most useful for prioritizing retention outreach, not for perfect certainty.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab2:
        lookup_id = st.text_input("Customer ID for prediction", value="")
        if lookup_id:
            row = rfm[rfm["CustomerID"] == lookup_id]
            if row.empty:
                st.warning("Customer not found.")
            else:
                r = row.iloc[0]
                sample = pd.DataFrame(
                    [{
                        "Frequency": r["Frequency"],
                        "Monetary": r["Monetary"],
                        "Avg_Order_Value": r["Avg_Order_Value"],
                        "Is_Frequent": r["Is_Frequent"],
                        "Is_High_Spender": r["Is_High_Spender"],
                    }]
                )
                pred = int(model.predict(sample)[0])
                prob = float(model.predict_proba(sample)[0][1]) if hasattr(model, "predict_proba") else float(pred)

                left, right = st.columns([1, 1])
                with left:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Frequency", int(r["Frequency"]))
                    c2.metric("Monetary", format_money(r["Monetary"]))
                    c3.metric("Avg Order", format_money(r["Avg_Order_Value"]))

                with right:
                    if pred == 1:
                        st.error(f"Predicted Churn • Probability: {prob:.2%}")
                    else:
                        st.success(f"Predicted Active • Churn Probability: {prob:.2%}")

                st.dataframe(
                    row[[
                        "CustomerID", "Recency", "Frequency", "Monetary",
                        "Segment", "Cluster_Label", "Churn"
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab3:
        left, right = st.columns(2)

        with left:
            cm = model_metrics["conf_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Active", "Actual Churn"],
                columns=["Pred Active", "Pred Churn"],
            )
            fig = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="YlOrBr",
                title="Confusion Matrix",
            )
            gold_template(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.bar(
                feature_importance_df.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance",
            )
            gold_template(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

        report_df = pd.DataFrame(model_metrics["report"]).T.reset_index().rename(columns={"index": "Class"})
        st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.markdown("### AI Narrative")
    st.markdown(
        """
        <div class="insight-box">
            <b>What this model does well:</b> It prioritizes churn-risk detection rather than chasing misleadingly high accuracy.
            In a retention context, recall matters because missed churners represent lost intervention opportunities.
        </div>
        <div class="insight-box">
            <b>How to use it:</b> Use churn probability to rank customers for outreach, discounts, or personalized offers.
            The model should support decisions, not replace judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# PAGE: INSIGHTS & STRATEGY
# =========================
elif page == "Insights & Strategy":
    add_hero(
        title="Strategic Findings and Recommended Actions",
        subtitle=(
            "This page translates analytics into business action across revenue growth, customer retention, "
            "inventory planning, and market expansion."
        ),
        right_text=(
            f"<div class='gold-label'>Recommended Focus</div>"
            f"<div style='font-size:1.35rem; font-weight:700; margin-top:8px;'>Retention + Upsell</div>"
            f"<div class='small-muted' style='margin-top:8px;'>Highest leverage based on model and segmentation</div>"
        ),
    )
    st.write("")

    monthly_summary = (
        filtered_sales_df.groupby("Month", as_index=False)["TotalPrice"]
        .sum()
        .sort_values("TotalPrice", ascending=False)
    )
    best_month = int(monthly_summary.iloc[0]["Month"]) if not monthly_summary.empty else None

    top_country = (
        filtered_sales_df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).index[0]
        if not filtered_sales_df.empty else "N/A"
    )

    top_product = (
        filtered_sales_df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).index[0]
        if not filtered_sales_df.empty else "N/A"
    )

    vip_count = int((rfm["Cluster_Label"] == "VIP").sum())
    at_risk_count = int((rfm["Cluster_Label"] == "At Risk").sum())

    left, right = st.columns(2)

    with left:
        st.markdown("### Key Findings")
        st.markdown(
            f"""
            <div class="insight-box"><b>1. Seasonal business:</b> The sales profile is seasonal, with strongest momentum clustering in late-year periods. Best month in current scope: <span class="gold-label">{best_month}</span>.</div>
            <div class="insight-box"><b>2. Market concentration:</b> The dominant market in current scope is <span class="gold-label">{top_country}</span>, indicating both strength and dependency.</div>
            <div class="insight-box"><b>3. Product concentration:</b> Top revenue product in current scope is <span class="gold-label">{top_product}</span>, reinforcing the need for inventory protection and cross-sell leverage.</div>
            <div class="insight-box"><b>4. Customer imbalance:</b> VIP customers are few (<span class="gold-label">{vip_count}</span>), while at-risk customers remain material (<span class="gold-label">{at_risk_count}</span>), creating clear retention urgency.</div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### Recommended Actions")
        st.markdown(
            """
            <div class="insight-box"><b>Defend VIPs:</b> Offer exclusivity, concierge support, early access, and personalized retention benefits.</div>
            <div class="insight-box"><b>Convert Regulars:</b> Use bundles, threshold discounts, and tailored recommendations to move regular customers toward higher frequency and spend.</div>
            <div class="insight-box"><b>Reactivate At-Risk:</b> Use churn scoring to prioritize win-back campaigns, timed incentives, and targeted email/SMS offers.</div>
            <div class="insight-box"><b>Plan for peak months:</b> Shift inventory, staffing, and marketing budgets ahead of demand spikes rather than reacting after they arrive.</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Executive Summary")
    st.markdown(
        """
        <div class="insight-box">
            The business exhibits a classic high-concentration retail structure: a relatively small set of customers and products drives a disproportionate share of value.
            The highest-return commercial strategy is not broad untargeted acquisition alone; it is a blend of defending premium customers,
            lifting regular customers upward, and intercepting churn before value disappears.
        </div>
        """,
        unsafe_allow_html=True,
    )

    export_cols = [
        "CustomerID", "Recency", "Frequency", "Monetary", "Avg_Order_Value",
        "Segment", "Cluster_Label", "Churn"
    ]
    csv_bytes = rfm[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Customer Intelligence CSV",
        data=csv_bytes,
        file_name="customer_intelligence.csv",
        mime="text/csv",
    )