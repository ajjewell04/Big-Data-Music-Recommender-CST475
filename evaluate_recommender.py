import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

CSV_PATH = "msd_flattened.csv"

FULL_FEATURE_COLS = ["tempo", "loudness", "duration"] + [f"timbre_mean_{i}" for i in range(12)]
EVAL_FEATURE_COLS = ["tempo", "loudness"]

K_VALUES = [1, 3, 5, 10]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["track_id", "title", "artist_name"] + FULL_FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = df[required].copy()

    for col in FULL_FEATURE_COLS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    before = len(clean)
    clean = clean.dropna(subset=FULL_FEATURE_COLS + ["title", "artist_name"]).reset_index(drop=True)
    after = len(clean)

    print(f"Loaded rows: {before}")
    print(f"Rows after dropping missing values: {after}")
    print(f"Dropped rows: {before - after}")
    print()

    return clean


def build_neighbor_graph(df: pd.DataFrame, feature_cols: list[str], max_k: int) -> dict:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    model = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    model.fit(X)

    distances, indices = model.kneighbors(X)

    return {
        "feature_cols": feature_cols,
        "neighbor_distances": distances[:, 1:max_k + 1],
        "neighbor_indices": indices[:, 1:max_k + 1],
    }


def evaluate_tempo_loudness_distance(
    df: pd.DataFrame,
    neighbor_indices: np.ndarray,
    k: int,
) -> dict:
    query_vals = df[EVAL_FEATURE_COLS].to_numpy(dtype=float)

    raw_diffs = []
    raw_tempo_abs = []
    raw_loudness_abs = []

    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df[EVAL_FEATURE_COLS])

    std_diffs = []
    std_tempo_abs = []
    std_loudness_abs = []

    for i in range(len(df)):
        nbr_idx = neighbor_indices[i, :k]

        raw_delta = query_vals[nbr_idx] - query_vals[i]
        std_delta = scaled_vals[nbr_idx] - scaled_vals[i]

        raw_diffs.append(raw_delta)
        std_diffs.append(std_delta)

        raw_tempo_abs.extend(np.abs(raw_delta[:, 0]))
        raw_loudness_abs.extend(np.abs(raw_delta[:, 1]))

        std_tempo_abs.extend(np.abs(std_delta[:, 0]))
        std_loudness_abs.extend(np.abs(std_delta[:, 1]))

    raw_diffs = np.vstack(raw_diffs)
    std_diffs = np.vstack(std_diffs)

    raw_euclidean = np.sqrt(np.sum(raw_diffs ** 2, axis=1))
    raw_manhattan = np.abs(raw_diffs[:, 0]) + np.abs(raw_diffs[:, 1])

    std_euclidean = np.sqrt(np.sum(std_diffs ** 2, axis=1))
    std_manhattan = np.abs(std_diffs[:, 0]) + np.abs(std_diffs[:, 1])

    return {
        "avg_euclidean_raw": float(np.mean(raw_euclidean)),
        "avg_manhattan_raw": float(np.mean(raw_manhattan)),
        "avg_abs_tempo_diff": float(np.mean(raw_tempo_abs)),
        "avg_abs_loudness_diff": float(np.mean(raw_loudness_abs)),
        "avg_euclidean_std": float(np.mean(std_euclidean)),
        "avg_manhattan_std": float(np.mean(std_manhattan)),
        "avg_abs_tempo_diff_std": float(np.mean(std_tempo_abs)),
        "avg_abs_loudness_diff_std": float(np.mean(std_loudness_abs)),
    }


def collect_metrics_by_k(df: pd.DataFrame, result: dict, label: str) -> pd.DataFrame:
    rows = []

    for k in K_VALUES:
        metrics = evaluate_tempo_loudness_distance(df, result["neighbor_indices"], k)
        metrics["k"] = k
        metrics["model"] = label
        rows.append(metrics)

    return pd.DataFrame(rows)


def print_metric_table(metrics_df: pd.DataFrame):
    print(metrics_df.to_string(index=False))
    print()


def plot_metrics(metrics_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Tempo/Loudness Evaluation Across k", fontsize=14)

    metric_specs = [
        ("avg_euclidean_std", "Avg Standardized Euclidean Distance"),
        ("avg_abs_tempo_diff", "Avg Absolute Tempo Difference"),
        ("avg_abs_loudness_diff", "Avg Absolute Loudness Difference"),
        ("avg_manhattan_std", "Avg Standardized Manhattan Distance"),
    ]

    models = metrics_df["model"].unique()

    for ax, (metric_col, title) in zip(axes.flat, metric_specs):
        for model in models:
            subset = metrics_df[metrics_df["model"] == model].sort_values("k")
            ax.plot(
                subset["k"],
                subset[metric_col],
                marker="o",
                linewidth=2,
                label=model,
            )

        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_ylabel(metric_col)
        ax.set_xticks(K_VALUES)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    df = load_data(CSV_PATH)
    max_k = max(K_VALUES)

    full_model_result = build_neighbor_graph(df, FULL_FEATURE_COLS, max_k)
    tempo_loudness_result = build_neighbor_graph(df, EVAL_FEATURE_COLS, max_k)

    full_metrics = collect_metrics_by_k(df, full_model_result, "Full-feature recommender")
    tl_metrics = collect_metrics_by_k(df, tempo_loudness_result, "Tempo+loudness recommender")

    metrics_df = pd.concat([full_metrics, tl_metrics], ignore_index=True)

    print("Dataset summary")
    print(f"Rows used for evaluation: {len(df)}")
    print(f"Evaluation features: {EVAL_FEATURE_COLS}")
    print()

    print_metric_table(metrics_df)
    plot_metrics(metrics_df)


if __name__ == "__main__":
    main()
