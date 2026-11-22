import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WEIGHT_KEYS = [
    ("WEIGHTS_FIT", ["mcq", "medcq"]),
    (
        "WEIGHTS_CONCENTRATION",
        ["rmcq", "circular_mad_mcq", "p_watson_dmcq", "p_sim_test_dmcq"],
    ),
    (
        "WEIGHTS_UNIFORMITY",
        [
            "mcd",
            "medcd",
            "rmcd",
            "circular_mad_mcd",
            "p_rayleigh_dmcd",
            "p_wilcoxon_dmcd",
        ],
    ),
    ("WEIGHTS_COMPOSITE", ["fit", "concentration", "uniformity"]),
]


def load_results(pattern="optuna_results_*.json"):
  files = sorted(glob.glob(pattern))
  if not files:
    raise FileNotFoundError(f"No files matched pattern: {pattern}")

  records = []
  feature_cols = [col for _, cols in WEIGHT_KEYS for col in cols]

  for path in files:
    with open(path, "r") as f:
      data = json.load(f)

    weights = data.get("weights", {})
    row = {
        "label": f"{data.get('dir', 'unknown')} | {data.get('measure', 'unknown')}",
        "dir": data.get("dir", "unknown"),
        "measure": data.get("measure", "unknown"),
        "rho": data.get("best_spearman_rho", np.nan),
    }

    for group_name, cols in WEIGHT_KEYS:
      vals = weights.get(group_name, [])
      for col, val in zip(cols, vals):
        row[col] = float(val)

    records.append(row)

  df = pd.DataFrame(records)
  return df, feature_cols


def plot_pca_clusters(df, pca, out_path="optuna_weights_pca_clusters.png"):
  fig, ax = plt.subplots(figsize=(9, 6.5))
  sns.scatterplot(
      data=df,
      x="PC1",
      y="PC2",
      hue="cluster",
      style="measure",
      s=140,
      ax=ax,
  )
  for _, r in df.iterrows():
    ax.annotate(
        r["label"],
        (r["PC1"], r["PC2"]),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=9,
    )
  var_pct = pca.explained_variance_ratio_ * 100
  ax.set_xlabel(
      f"PC1 ({var_pct[0]:.1f}% var)" if len(var_pct) > 0 else "PC1"
  )
  ax.set_ylabel(
      f"PC2 ({var_pct[1]:.1f}% var)" if len(var_pct) > 1 else "PC2 (0.0%)"
  )
  ax.set_title("PCA of Tuned Weights (Colored by Cluster)")
  ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

  plt.tight_layout()
  plt.savefig(out_path, dpi=300, bbox_inches="tight")
  plt.show()
  plt.close(fig)


def plot_weight_heatmap(df, feature_cols, out_path="optuna_weights_heatmap.png"):
  fig, ax = plt.subplots(figsize=(11, 6.5))
  heatmap_df = df.set_index("label")[feature_cols]
  sns.heatmap(
      heatmap_df,
      cmap="YlGnBu",
      vmin=0.0,
      vmax=1.0,
      linewidths=0.5,
      cbar_kws={"label": "Normalized Weight"},
      ax=ax,
  )
  ax.set_title("Weight Profiles Across Runs")
  ax.set_xlabel("Score Weight")
  ax.set_ylabel("")
  ax.tick_params(axis="x", rotation=45)

  plt.tight_layout()
  plt.savefig(out_path, dpi=300, bbox_inches="tight")
  plt.show()
  plt.close(fig)


def main():
  df, feature_cols = load_results("optuna_results_*.json")

  if len(df) < 2:
    print("Need at least 2 Optuna result JSON files to cluster and run PCA.")
    return

  X = df[feature_cols].to_numpy()
  X_scaled = StandardScaler().fit_transform(X)

  n_components = min(2, X_scaled.shape[0], X_scaled.shape[1])
  pca = PCA(n_components=n_components)
  coords = pca.fit_transform(X_scaled)

  df["PC1"] = coords[:, 0]
  df["PC2"] = coords[:, 1] if n_components > 1 else 0.0

  n_clusters = min(3, len(df))
  clustering = AgglomerativeClustering(n_clusters=n_clusters)
  df["cluster"] = clustering.fit_predict(X_scaled).astype(str)

  loadings = pd.DataFrame(
      pca.components_.T,
      index=feature_cols,
      columns=[f"PC{i+1}" for i in range(n_components)],
  )
  print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
  print("\nTop loadings by absolute weight on PC1:")
  print(
      loadings["PC1"]
      .reindex(loadings["PC1"].abs().sort_values(ascending=False).index)
      .round(3)
  )

  sns.set_theme(style="whitegrid", context="talk")
  plot_pca_clusters(df, pca, "optuna_weights_pca_clusters.png")
  plot_weight_heatmap(df, feature_cols, "optuna_weights_heatmap.png")


if __name__ == "__main__":
  main()