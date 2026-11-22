import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

ORIGINAL_WEIGHTS = {
    "mcq": 0.4,
    "medcq": 0.6,
    "rmcq": 0.15,
    "circular_mad_mcq": 0.3,
    "p_watson_dmcq": 0.25,
    "p_sim_test_dmcq": 0.3,
    "mcd": 0.1,
    "medcd": 0.2,
    "rmcd": 0.1,
    "circular_mad_mcd": 0.2,
    "p_rayleigh_dmcd": 0.15,
    "p_wilcoxon_dmcd": 0.25,
    "fit": 0.5,
    "concentration": 0.2,
    "uniformity": 0.3,
}


def load_weights_df(pattern="optuna_results_*.json"):
  files = sorted(glob.glob(pattern))
  if not files:
    raise FileNotFoundError(f"No files matched pattern: {pattern}")

  feature_cols = [col for _, cols in WEIGHT_KEYS for col in cols]
  rows = []

  for path in files:
    with open(path, "r") as f:
      data = json.load(f)

    weights = data.get("weights", {})
    record = {
        "run": f"{data.get('dir', 'unknown')} | {data.get('measure', 'unknown')}",
        "dir": data.get("dir", "unknown"),
        "measure": data.get("measure", "unknown"),
    }

    for group_name, cols in WEIGHT_KEYS:
      vals = weights.get(group_name, [])
      for col, val in zip(cols, vals):
        record[col] = float(val)

    rows.append(record)

  df = pd.DataFrame(rows)
  long_df = df.melt(
      id_vars=["run", "dir", "measure"],
      value_vars=feature_cols,
      var_name="weight",
      value_name="value",
  )
  return df, long_df, feature_cols


def main():
  df, long_df, feature_cols = load_weights_df("optuna_results_*.json")

  summary = (
      long_df.groupby("weight", sort=False)["value"]
      .agg(mean="mean", median="median", std="std", count="count")
      .reindex(feature_cols)
  )
  summary["original"] = summary.index.map(ORIGINAL_WEIGHTS)
  print(summary[["original", "mean", "median", "std", "count"]].round(4))

  plot_df = summary.reset_index().melt(
      id_vars="weight",
      value_vars=["mean", "median"],
      var_name="statistic",
      value_name="value",
  )

  sns.set_theme(style="whitegrid", context="talk")
  fig, ax = plt.subplots(figsize=(12, 7))

  sns.stripplot(
      data=long_df,
      y="weight",
      x="value",
      order=feature_cols,
      color="gray",
      alpha=0.45,
      size=6,
      jitter=0.15,
      ax=ax,
  )

  sns.pointplot(
      data=plot_df,
      y="weight",
      x="value",
      hue="statistic",
      order=feature_cols,
      dodge=0.25,
      linestyles="none",
      markers=["o", "D"],
      palette={"mean": "#1f77b4", "median": "#d95f02"},
      ax=ax,
  )

  orig_vals = [ORIGINAL_WEIGHTS[col] for col in feature_cols]
  y_positions = np.arange(len(feature_cols))
  ax.scatter(
      orig_vals,
      y_positions,
      color="#2ca02c",
      marker="X",
      s=120,
      zorder=6,
      label="original",
  )

  ax.set_xlim(-0.02, 1.02)
  ax.set_xlabel("Normalized Weight Value")
  ax.set_ylabel("")
  ax.set_title(
      f"Mean vs. Median Tuned Weights vs. Original Across {len(df)} Runs"
  )

  # Combine legend entries from pointplot and scatter cleanly
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(
      handles,
      labels,
      title="Summary",
      bbox_to_anchor=(1.02, 1),
      loc="upper left",
  )

  plt.tight_layout()
  plt.savefig("optuna_weights_mean_median.png", dpi=300, bbox_inches="tight")
  plt.show()


if __name__ == "__main__":
  main()