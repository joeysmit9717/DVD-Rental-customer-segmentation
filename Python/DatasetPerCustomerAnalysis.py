"""
KMeans clustering analysis of the per-customer dataset.

The dataset (DatasetPerCustomerTable.csv) contains one row per customer with:
    - customer_id : unique customer identifier (not used as a feature)
    - average     : average spend per purchase (per movie watched)
    - total       : total amount spent by the customer
    - <genres>    : proportion of the customer's rentals in each genre
                    (action, animation, children, ... travel)

The script standardizes the features, picks a sensible number of clusters
using the silhouette score, fits a final KMeans model, prints a summary of
the clusters and displays a few diagnostic plots.

It also compares KMeans against Gaussian Mixture and Agglomerative (Ward)
clustering to justify the chosen number of clusters, and produces a
standardized cluster-profile heatmap, a dendrogram and a per-sample
silhouette diagram to help interpret and validate the segmentation.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# Resolve paths relative to this file so the script works from any cwd.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "DatasetPerCustomerTable.csv")
# Portfolio export folder (README / LaTeX / PNGs live here).
PORTFOLIO_DIR = os.path.join(
    BASE_DIR, "portfolio", "DatasetpercustomerPlotsAndLatex"
)
FIGURES_DIR = os.path.join(PORTFOLIO_DIR, "figures")

# Range of cluster counts to evaluate when searching for the best k.
K_RANGE = range(2, 20)
RANDOM_STATE = 42
# Set SAVE_FIGURES=0 to skip writing PNGs (interactive exploration only).
SAVE_FIGURES = os.environ.get("SAVE_FIGURES", "1") != "0"


def save_figure(filename):
    """Save the current matplotlib figure into the portfolio figures folder."""
    if not SAVE_FIGURES:
        return
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")

def cluster_cmap_name(n_clusters):
    """Pick a qualitative colormap with enough distinct colours for n clusters.

    ``tab10`` only has 10 colours, so with more clusters the colours would
    repeat and become impossible to tell apart. ``tab20`` covers up to 20.
    """
    return "tab10" if n_clusters <= 10 else "tab20"


def load_data(path):
    """Load the CSV and split it into the id column and the feature matrix."""
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != "customer_id"]
    return df, feature_cols


def find_best_k(X_scaled, k_range):
    """Evaluate KMeans for each k and return inertias and silhouette scores."""
    inertias = {}
    silhouettes = {}
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertias[k] = model.inertia_
        silhouettes[k] = silhouette_score(X_scaled, labels)
    best_k = max(silhouettes, key=silhouettes.get)
    return inertias, silhouettes, best_k


def plot_diagnostics(inertias, silhouettes):
    """Plot the elbow curve and silhouette scores side by side."""
    ks = list(inertias.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ks, [inertias[k] for k in ks], "o-")
    axes[0].set_title("Elbow method (inertia)")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(ks, [silhouettes[k] for k in ks], "o-", color="tab:green")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")

    fig.tight_layout()


def plot_clusters_pca(X_scaled, labels, centers_scaled):
    """Project the data to 2D with PCA and plot the clusters."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)
    centers_2d = pca.transform(centers_scaled)

    cmap_name = cluster_cmap_name(len(np.unique(labels)))
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1], c=labels, cmap=cmap_name, s=25, alpha=0.7
    )
    # Colour each centroid like its own cluster, with a black outline so the
    # crosses stay visible on top of the points.
    colors = cluster_colors(labels)
    center_colors = [colors[c] for c in range(len(centers_2d))]
    ax.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c=center_colors,
        marker="X",
        s=220,
        edgecolors="black",
        linewidths=1.5,
        label="Centroids",
    )
    var = pca.explained_variance_ratio_
    ax.set_title("Customer clusters (PCA projection)")
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}% variance)")
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(ax.legend(loc="upper right"))

    fig.tight_layout()


def cluster_colors(labels):
    """Return a dict mapping each cluster label to its scatter-plot colour.

    This reproduces exactly how ``ax.scatter(..., c=labels, cmap=...)``
    assigns colours: the integer labels are normalised across their min/max
    range and then looked up in the colormap.
    """
    unique = np.unique(labels)
    cmap = plt.get_cmap(cluster_cmap_name(len(unique)))
    norm = Normalize(vmin=labels.min(), vmax=labels.max())
    return {int(c): cmap(norm(c)) for c in unique}


def describe_clusters(df, feature_cols):
    """Build a short, human-readable description for each cluster.

    For every cluster we report size, how total spend compares to the overall
    customer base, and the genres the cluster watches most.
    """
    genre_cols = [c for c in feature_cols if c not in ("average", "total")]

    overall = df[feature_cols].mean()
    std = df[feature_cols].std()

    def level(value, mean, sd):
        """Classify a value as low / average / high relative to the population."""
        if sd == 0:
            return "average"
        if value > mean + 0.5 * sd:
            return "high"
        if value < mean - 0.5 * sd:
            return "low"
        return "average"

    rows = []
    profile = df.groupby("cluster")[feature_cols].mean()
    sizes = df["cluster"].value_counts()
    total_customers = len(df)

    for cluster in sorted(df["cluster"].unique()):
        means = profile.loc[cluster]
        size = int(sizes.loc[cluster])
        pct = 100 * size / total_customers

        spend_level = level(means["total"], overall["total"], std["total"])

        top_genres = means[genre_cols].sort_values(ascending=False).head(3)
        genre_text = ", ".join(
            f"{g} ({share * 100:.0f}%)" for g, share in top_genres.items()
        )

        description = (
            f"{size} customers ({pct:.0f}%). "
            f"{spend_level.capitalize()} total spend (avg ${means['total']:.0f}). "
            f"Favourite genres: {genre_text}."
        )
        rows.append((int(cluster), description))

    return rows


def plot_cluster_legend_table(df, feature_cols, labels):
    """Draw a table that explains what each coloured cluster represents."""
    colors = cluster_colors(labels)
    rows = describe_clusters(df, feature_cols)

    fig, ax = plt.subplots(figsize=(13, 0.7 * len(rows) + 1.5))
    ax.axis("off")
    ax.set_title("What each cluster colour means", fontsize=13, pad=12)

    table = ax.table(
        cellText=[["", str(cluster), desc] for cluster, desc in rows],
        colLabels=["Colour", "Cluster", "Description"],
        colWidths=[0.07, 0.08, 0.85],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Fill the "Colour" cell of each data row with the matching cluster colour.
    for i, (cluster, _) in enumerate(rows):
        swatch = table[i + 1, 0]  # +1 to skip the header row
        swatch.set_facecolor(colors[cluster])

    # Make the header row bold.
    for col in range(3):
        table[0, col].set_text_props(fontweight="bold")

    fig.tight_layout()


def plot_cluster_profile_heatmap(df, feature_cols):
    """Heatmap of cluster means expressed in population standard deviations.

    Each feature is first standardized across the whole customer base, then
    averaged per cluster. A cell value of +1.5 means that cluster sits 1.5
    standard deviations above the average customer on that feature. Red marks
    above-average values, blue below-average, so the colours read directly as
    "what defines this cluster". This is the standard, rigorous way to profile
    segments (more informative than listing each cluster's top genres).
    """
    overall_mean = df[feature_cols].mean()
    overall_std = df[feature_cols].std().replace(0, 1)
    z = (df[feature_cols] - overall_mean) / overall_std
    z["cluster"] = df["cluster"].values
    profile = z.groupby("cluster")[feature_cols].mean()

    clusters = profile.index.tolist()
    data = profile.to_numpy()
    # Symmetric colour range centred on 0 so red/blue are comparable.
    vmax = np.abs(data).max()

    fig, ax = plt.subplots(
        figsize=(0.6 * len(feature_cols) + 3, 0.5 * len(clusters) + 2)
    )
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {c}" for c in clusters])
    ax.set_title("Cluster profiles (standard deviations from population average)")

    for i in range(len(clusters)):
        for j in range(len(feature_cols)):
            val = data[i, j]
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                color="white" if abs(val) > vmax * 0.5 else "black",
                fontsize=7,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Std. deviations from average")
    fig.tight_layout()


def compare_clustering_algorithms(X_scaled, k_range):
    """Compare silhouette scores of three algorithms across a range of k.

    Plotting KMeans, Gaussian Mixture and Agglomerative (Ward) on the same axes
    is a much stronger justification for the number of clusters than trusting a
    single algorithm: if very different methods agree on where the score peaks,
    that's real structure; if they all stay low, the data simply doesn't split
    into tight clusters.
    """
    results = {"KMeans": {}, "Gaussian Mixture": {}, "Agglomerative (Ward)": {}}
    for k in k_range:
        km = KMeans(
            n_clusters=k, random_state=RANDOM_STATE, n_init=10
        ).fit_predict(X_scaled)
        results["KMeans"][k] = silhouette_score(X_scaled, km)

        gmm = GaussianMixture(
            n_components=k, random_state=RANDOM_STATE
        ).fit_predict(X_scaled)
        results["Gaussian Mixture"][k] = silhouette_score(X_scaled, gmm)

        agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
            X_scaled
        )
        results["Agglomerative (Ward)"][k] = silhouette_score(X_scaled, agg)

    ks = list(k_range)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, scores in results.items():
        ax.plot(ks, [scores[k] for k in ks], "o-", label=name)
    ax.set_title("Silhouette score by algorithm")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score (higher is better)")
    ax.legend()
    fig.tight_layout()
    return results


def plot_gmm_model_selection(X_scaled, k_range):
    """Plot BIC and AIC for Gaussian Mixtures and return the best k by BIC.

    Unlike KMeans, a Gaussian Mixture is a probabilistic model, so we can use
    likelihood-based criteria. BIC and AIC reward goodness of fit but penalise
    extra components; the k with the lowest BIC is the principled choice.
    """
    bic, aic = {}, {}
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE).fit(
            X_scaled
        )
        bic[k] = gmm.bic(X_scaled)
        aic[k] = gmm.aic(X_scaled)
    best_k = min(bic, key=bic.get)

    ks = list(k_range)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, [bic[k] for k in ks], "o-", label="BIC")
    ax.plot(ks, [aic[k] for k in ks], "o-", label="AIC")
    ax.axvline(
        best_k, color="grey", linestyle="--", label=f"Lowest BIC (k={best_k})"
    )
    ax.set_title("Gaussian Mixture model selection")
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Information criterion (lower is better)")
    ax.legend()
    fig.tight_layout()
    return best_k


def plot_dendrogram(X_scaled, max_leaves=30):
    """Plot a Ward-linkage dendrogram of the customers.

    A dendrogram shows how customers progressively merge into larger groups;
    the height of each merge is the distance between the groups joined. A large
    vertical gap before a merge is a natural place to 'cut' the tree, which
    suggests how many clusters the data really supports. The tree is truncated
    to its last ``max_leaves`` merges so it stays readable.
    """
    linkage_matrix = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(11, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=max_leaves,
        leaf_rotation=90.0,
        ax=ax,
    )
    ax.set_title(
        f"Hierarchical clustering dendrogram (Ward, last {max_leaves} merges)"
    )
    ax.set_xlabel("Customer groups (leaf = number of customers in parentheses)")
    ax.set_ylabel("Merge distance")
    fig.tight_layout()


def plot_silhouette_diagram(X_scaled, labels, title):
    """Per-sample silhouette ('knife') plot for a fitted clustering.

    Every customer gets a silhouette coefficient in [-1, 1]: high means it sits
    comfortably inside its own cluster, negative means it probably belongs to a
    neighbour. Bars are grouped and sorted per cluster, so wide blocks reaching
    right are healthy clusters, while thin blocks or bars crossing zero flag
    weak or overlapping ones. The dashed line is the overall average.
    """
    sample_silhouette = silhouette_samples(X_scaled, labels)
    avg = silhouette_score(X_scaled, labels)
    clusters = np.unique(labels)
    cmap = plt.get_cmap(cluster_cmap_name(len(clusters)))
    norm = Normalize(vmin=labels.min(), vmax=labels.max())

    fig, ax = plt.subplots(figsize=(9, 7))
    y_lower = 10
    for c in clusters:
        values = np.sort(sample_silhouette[labels == c])
        size = len(values)
        y_upper = y_lower + size
        color = cmap(norm(c))
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            values,
            facecolor=color,
            edgecolor=color,
            alpha=0.8,
        )
        ax.text(-0.05, y_lower + 0.5 * size, str(c))
        y_lower = y_upper + 10

    ax.axvline(avg, color="red", linestyle="--", label=f"Average = {avg:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])
    ax.legend(loc="lower right")
    fig.tight_layout()


def main():
    df, feature_cols = load_data(DATA_PATH)
    print(f"Loaded {len(df)} customers with {len(feature_cols)} features.")

    X = df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias, silhouettes, best_k = find_best_k(X_scaled, K_RANGE)

    print("\nSilhouette scores by k:")
    for k in K_RANGE:
        marker = "  <-- best" if k == best_k else ""
        print(f"  k={k}: silhouette={silhouettes[k]:.3f}{marker}")
    print(f"\nChosen number of clusters: k={best_k}")

    plot_diagnostics(inertias, silhouettes)
    save_figure("elbow_silhouette.png")

    # Cross-check the choice of k against other clustering algorithms.
    compare_clustering_algorithms(X_scaled, K_RANGE)
    save_figure("algorithm_comparison.png")
    gmm_best_k = plot_gmm_model_selection(X_scaled, K_RANGE)
    save_figure("gmm_bic_aic.png")
    print(f"Gaussian Mixture suggests k={gmm_best_k} (lowest BIC).")
    plot_dendrogram(X_scaled)
    save_figure("dendrogram.png")

    final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = final_model.fit_predict(X_scaled)
    df["cluster"] = labels

    plot_clusters_pca(X_scaled, labels, final_model.cluster_centers_)
    save_figure("pca_clusters.png")
    plot_cluster_legend_table(df, feature_cols, labels)
    save_figure("cluster_legend.png")
    plot_cluster_profile_heatmap(df, feature_cols)
    save_figure("cluster_heatmap.png")
    plot_silhouette_diagram(
        X_scaled, labels, f"Silhouette diagram (KMeans, k={best_k})"
    )
    save_figure("silhouette_best_k.png")

    # Contrast against a smaller, simpler clustering to show how fewer, larger
    # clusters change the silhouette shape (fatter, longer wedges).
    labels_k5 = KMeans(
        n_clusters=5, random_state=RANDOM_STATE, n_init=10
    ).fit_predict(X_scaled)
    plot_silhouette_diagram(X_scaled, labels_k5, "Silhouette diagram (KMeans, k=5)")
    save_figure("silhouette_k5.png")

    # Cluster sizes and the mean of each feature per cluster (in original units).
    print("\nCluster sizes:")
    print(df["cluster"].value_counts().sort_index().to_string())

    profile = df.groupby("cluster")[feature_cols].mean()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\nCluster profiles (mean of each feature):")
    print(profile.round(3).to_string())

    print("\nCluster descriptions:")
    for cluster, desc in describe_clusters(df, feature_cols):
        print(f"  Cluster {cluster}: {desc}")

    # Display the diagnostic and cluster plots interactively (skip in headless).
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
