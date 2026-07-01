import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATA_DIR = Path("data")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "crimes_raw.csv")
    # Drop rows with missing coordinates (rare but possible)
    df = df.dropna(subset=["latitude", "longitude"])
    print(f"Loaded {len(df)} records with valid coordinates")
    return df


def find_optimal_k(coords_scaled: np.ndarray, k_range: range) -> dict:
    """
    Use two methods to find the best number of clusters:
    1. Elbow method (inertia)
    2. Silhouette score
    """
    inertias = []
    silhouette_scores = []

    print("\nFinding optimal K...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords_scaled)
        inertias.append(kmeans.inertia_)

        # Silhouette needs at least 2 clusters
        if k >= 2:
            score = silhouette_score(coords_scaled, labels, sample_size=5000)
            silhouette_scores.append(score)
            print(f"  K={k} | inertia={kmeans.inertia_:.0f} | silhouette={score:.4f}")
        else:
            silhouette_scores.append(None)
            print(f"  K={k} | inertia={kmeans.inertia_:.0f} | silhouette=N/A")

    return {"inertias": inertias, "silhouette_scores": silhouette_scores}


def plot_elbow_and_silhouette(k_range: range, results: dict):
    """Plot elbow curve and silhouette scores side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow
    ax1.plot(list(k_range), results["inertias"], marker="o", color="steelblue", linewidth=2)
    ax1.set_title("Elbow Method — Inertia vs K", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Silhouette
    scores = [s for s in results["silhouette_scores"] if s is not None]
    k_values = [k for k, s in zip(k_range, results["silhouette_scores"]) if s is not None]
    best_k = k_values[scores.index(max(scores))]

    ax2.plot(k_values, scores, marker="o", color="coral", linewidth=2)
    ax2.axvline(x=best_k, color="green", linestyle="--", alpha=0.7, label=f"Best K={best_k}")
    ax2.set_title("Silhouette Score vs K", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Number of clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(DATA_DIR / "chart_elbow_silhouette.png", dpi=150)
    plt.show()
    print(f"\nBest K by silhouette score: {best_k}")
    return best_k


def run_final_clustering(df: pd.DataFrame, coords_scaled: np.ndarray, k: int) -> pd.DataFrame:
    """Run KMeans with the chosen K and add cluster labels to the dataframe."""
    print(f"\nRunning final KMeans with K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(coords_scaled)

    # Cluster summary
    summary = df.groupby("cluster").agg(
        crime_count=("id", "count"),
        top_category=("category", lambda x: x.value_counts().index[0]),
        center_lat=("latitude", "mean"),
        center_lng=("longitude", "mean"),
    ).reset_index()

    print("\nCluster Summary:")
    print(summary.to_string(index=False))

    return df, kmeans, summary


def plot_clusters(df: pd.DataFrame, k: int):
    """Plot each cluster in a different colour."""
    colors = plt.colormaps["tab10"].resampled(k)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cluster_id in range(k):
        subset = df[df["cluster"] == cluster_id]
        ax.scatter(
            subset["longitude"], subset["latitude"],
            alpha=0.3, s=6,
            color=colors(cluster_id),
            label=f"Cluster {cluster_id} (n={len(subset)})"
        )

    ax.set_title(f"Crime Hotspot Clusters (K={k}) — Leeds 2024",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", markerscale=3)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "chart_clusters.png", dpi=150)
    plt.show()


def main():
    # 1. Load
    df = load_data()

    # 2. Extract coordinates and scale
    coords = df[["latitude", "longitude"]].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # 3. Find optimal K (test K=2 to K=10)
    k_range = range(2, 11)
    results = find_optimal_k(coords_scaled, k_range)
    best_k = plot_elbow_and_silhouette(k_range, results)
    best_k = 5 
    
    # 4. Run final clustering with best K
    df_clustered, kmeans, summary = run_final_clustering(df, coords_scaled, best_k)

    # 5. Plot clusters
    plot_clusters(df_clustered, best_k)

    # 6. Save clustered data for the dashboard
    output_path = DATA_DIR / "crimes_clustered.csv"
    df_clustered.to_csv(output_path, index=False)
    print(f"\nSaved clustered data to {output_path}")


if __name__ == "__main__":
    main()