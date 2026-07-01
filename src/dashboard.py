import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK Crime Explorer — Leeds 2024",
    page_icon="🔍",
    layout="wide",
)

DATA_PATH = Path("data/crimes_clustered.csv")

LAT = 53.7996
LNG = -1.5491
MONTHS = [
    "2024-01", "2024-02", "2024-03", "2024-04",
    "2024-05", "2024-06", "2024-07", "2024-08",
    "2024-09", "2024-10", "2024-11", "2024-12",
]
BASE_URL = "https://data.police.uk/api/crimes-street/all-crime"


def fetch_and_cluster() -> pd.DataFrame:
    """Fetch from API and run clustering — used when CSV doesn't exist."""
    import requests
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    all_records = []
    for month in MONTHS:
        r = requests.get(BASE_URL, params={"lat": LAT, "lng": LNG, "date": month}, timeout=30)
        if r.status_code == 200:
            all_records.extend(r.json())

    df = pd.DataFrame(all_records)

    # Flatten nested columns
    df["latitude"] = df["location"].apply(lambda x: float(x["latitude"]) if x else None)
    df["longitude"] = df["location"].apply(lambda x: float(x["longitude"]) if x else None)
    df["street_name"] = df["location"].apply(
        lambda x: x["street"]["name"] if x and "street" in x else None
    )
    df["outcome_category"] = df["outcome_status"].apply(
        lambda x: x["category"] if x else "Unknown"
    )
    df = df.drop(columns=["location", "outcome_status", "context"], errors="ignore")
    df = df.dropna(subset=["latitude", "longitude"])

    # Cluster
    coords_scaled = StandardScaler().fit_transform(df[["latitude", "longitude"]].values)
    df["cluster"] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(coords_scaled)

    # Save for next time
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    return df

# ── Cluster colour palette (matches map markers) ──────────────────────────────
CLUSTER_COLOURS = {
    0: "#2196F3",  # blue
    1: "#F44336",  # red
    2: "#4CAF50",  # green
    3: "#FF9800",  # orange
    4: "#9C27B0",  # purple
}


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        with st.spinner("Fetching live data from UK Police API... (first load only, ~30 seconds)"):
            df = fetch_and_cluster()
    df["month"] = pd.to_datetime(df["month"])
    return df


df_full = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Flag_of_the_United_Kingdom.svg/320px-Flag_of_the_United_Kingdom.svg.png",
    width=120,
)
st.sidebar.title("Filters")

# Crime category filter
all_categories = sorted(df_full["category"].unique())
selected_categories = st.sidebar.multiselect(
    "Crime category",
    options=all_categories,
    default=all_categories,
)

# Month range filter
months = sorted(df_full["month"].dt.strftime("%Y-%m").unique())
month_start, month_end = st.sidebar.select_slider(
    "Month range",
    options=months,
    value=(months[0], months[-1]),
)

# Cluster filter
all_clusters = sorted(df_full["cluster"].unique())
selected_clusters = st.sidebar.multiselect(
    "Hotspot cluster",
    options=all_clusters,
    default=all_clusters,
    format_func=lambda x: f"Cluster {x}",
)

# ── Apply filters ─────────────────────────────────────────────────────────────
mask = (
    df_full["category"].isin(selected_categories)
    & (df_full["month"].dt.strftime("%Y-%m") >= month_start)
    & (df_full["month"].dt.strftime("%Y-%m") <= month_end)
    & df_full["cluster"].isin(selected_clusters)
)
df = df_full[mask].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 UK Crime Explorer — Leeds 2024")
st.markdown(
    "Interactive explorer of street-level crime data from the "
    "[UK Police API](https://data.police.uk). "
    "Use the sidebar to filter by category, month, and hotspot cluster."
)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total crimes", f"{len(df):,}")
k2.metric("Crime categories", df["category"].nunique())
k3.metric("Streets affected", df["street_name"].nunique())
top_cat = df["category"].value_counts().idxmax() if len(df) > 0 else "—"
k4.metric("Top category", top_cat.replace("-", " ").title())

st.divider()

# ── Map + cluster breakdown ───────────────────────────────────────────────────
col_map, col_cluster = st.columns([2, 1])

with col_map:
    st.subheader("📍 Crime hotspot map")

    # Build folium map
    centre_lat = df["latitude"].mean() if len(df) > 0 else 53.7996
    centre_lng = df["longitude"].mean() if len(df) > 0 else -1.5491

    m = folium.Map(location=[centre_lat, centre_lng], zoom_start=13, tiles="CartoDB positron")

    # Sample up to 3000 points for performance
    df_sample = df.sample(min(3000, len(df)), random_state=42) if len(df) > 0 else df

    for _, row in df_sample.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=CLUSTER_COLOURS.get(row["cluster"], "#999"),
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>{row['category'].replace('-', ' ').title()}</b><br>"
                f"{row['street_name']}<br>"
                f"{row['month'].strftime('%B %Y')}<br>"
                f"Outcome: {row['outcome_category']}",
                max_width=200,
            ),
        ).add_to(m)

    st_folium(m, width=700, height=450)

with col_cluster:
    st.subheader("🗂 Cluster breakdown")

    cluster_summary = (
        df.groupby("cluster")
        .agg(
            count=("id", "count"),
            top_crime=("category", lambda x: x.value_counts().index[0]),
        )
        .reset_index()
    )

    for _, row in cluster_summary.iterrows():
        colour = CLUSTER_COLOURS.get(row["cluster"], "#999")
        st.markdown(
            f"""
            <div style="border-left: 4px solid {colour}; padding: 8px 12px;
                        margin-bottom: 10px; border-radius: 4px;
                        background: #f9f9f9;">
                <b>Cluster {int(row['cluster'])}</b><br>
                <span style="font-size:13px">{int(row['count']):,} crimes</span><br>
                <span style="font-size:12px; color:#666">
                    Top: {row['top_crime'].replace('-',' ').title()}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Charts row ────────────────────────────────────────────────────────────────
col_trend, col_cat = st.columns(2)

with col_trend:
    st.subheader("📈 Monthly crime trend")
    monthly = df.groupby(df["month"].dt.strftime("%b %Y")).size().reset_index(name="count")
    monthly.columns = ["month", "count"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(monthly["month"], monthly["count"], marker="o", color="#2196F3", linewidth=2)
    ax.fill_between(range(len(monthly)), monthly["count"], alpha=0.1, color="#2196F3")
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["month"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Incidents")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_cat:
    st.subheader("🏷 Crime by category")
    cat_counts = df["category"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    cat_counts.plot(kind="barh", ax=ax, color="#F44336")
    ax.set_xlabel("Incidents")
    ax.invert_yaxis()
    ax.set_yticklabels(
        [c.replace("-", " ").title() for c in cat_counts.index], fontsize=9
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Top streets table ─────────────────────────────────────────────────────────
st.subheader("🛣 Top streets by crime count")
top_streets = (
    df.groupby("street_name")
    .agg(count=("id", "count"), top_crime=("category", lambda x: x.value_counts().index[0]))
    .sort_values("count", ascending=False)
    .head(15)
    .reset_index()
)
top_streets.columns = ["Street", "Crime Count", "Most Common Crime"]
top_streets["Most Common Crime"] = top_streets["Most Common Crime"].str.replace("-", " ").str.title()
st.dataframe(top_streets, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Data source: [data.police.uk](https://data.police.uk) · "
    "Built with Python, scikit-learn, and Streamlit · "
    "Part of UK Crime Explorer portfolio project</small>",
    unsafe_allow_html=True,
)