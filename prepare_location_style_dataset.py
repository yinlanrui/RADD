import argparse
import gzip
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def _open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def load_snap_checkins(path):
    rows = []
    with _open_maybe_gzip(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            try:
                rows.append((parts[0], parts[4], float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["user", "feature", "lat", "lon"])


def load_foursquare_checkins(path):
    rows = []
    with _open_maybe_gzip(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            try:
                rows.append((parts[0], parts[1], float(parts[4]), float(parts[5])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["user", "feature", "lat", "lon"])


def build_location_style_npz(
        raw_path, source_format, output_dir, output_name,
        feature_dim=446, num_classes=30, min_user_checkins=10,
        max_users=30000, seed=7):
    if source_format == "snap":
        df = load_snap_checkins(raw_path)
    elif source_format == "foursquare":
        df = load_foursquare_checkins(raw_path)
    else:
        raise ValueError(f"Unsupported source_format: {source_format}")

    if df.empty:
        raise RuntimeError(f"No valid check-ins were parsed from {raw_path}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon"])
    user_counts = df["user"].value_counts()
    keep_users = user_counts[user_counts >= min_user_checkins].index
    df = df[df["user"].isin(keep_users)]
    if df.empty:
        raise RuntimeError("No users remain after min_user_checkins filtering")

    user_stats = df.groupby("user").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("feature", "size"))
    if max_users and len(user_stats) > max_users:
        user_stats = user_stats.sample(n=max_users, random_state=seed)
        df = df[df["user"].isin(user_stats.index)]

    top_features = df["feature"].value_counts().head(feature_dim).index.tolist()
    if len(top_features) < feature_dim:
        raise RuntimeError(
            f"Need at least {feature_dim} distinct features, found {len(top_features)}")

    df = df[df["feature"].isin(top_features)]
    users = user_stats.index.astype(str).tolist()
    user_to_row = {user: idx for idx, user in enumerate(users)}
    feature_to_col = {feature: idx for idx, feature in enumerate(top_features)}

    x = np.zeros((len(users), feature_dim), dtype=np.float32)
    for user, feature in zip(df["user"].astype(str), df["feature"].astype(str)):
        row = user_to_row.get(user)
        col = feature_to_col.get(feature)
        if row is not None and col is not None:
            x[row, col] = 1.0

    nonempty = x.sum(axis=1) > 0
    users = [u for u, keep in zip(users, nonempty) if keep]
    x = x[nonempty]
    coords = user_stats.loc[users, ["lat", "lon"]].to_numpy(dtype=np.float32)

    if len(users) < num_classes * 5:
        raise RuntimeError(
            f"Too few users ({len(users)}) for {num_classes} classes after preprocessing")

    kmeans = MiniBatchKMeans(
        n_clusters=num_classes,
        random_state=seed,
        batch_size=min(4096, max(256, len(users))),
        n_init=10)
    y = kmeans.fit_predict(coords).astype(np.int64)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "data_complete.npz")
    np.savez_compressed(out_path, x=x, y=y)

    with open(os.path.join(output_dir, "feature_vocab.txt"), "w", encoding="utf-8") as f:
        for feature in top_features:
            f.write(f"{feature}\n")

    class_counts = Counter(y.tolist())
    pd.DataFrame({
        "class_id": sorted(class_counts),
        "count": [class_counts[i] for i in sorted(class_counts)],
    }).to_csv(os.path.join(output_dir, "class_counts.csv"), index=False)

    metadata = {
        "name": output_name,
        "source_format": source_format,
        "raw_path": raw_path,
        "num_users": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "num_classes": int(num_classes),
        "min_user_checkins": int(min_user_checkins),
        "max_users": int(max_users),
        "seed": int(seed),
        "class_min": int(min(class_counts.values())),
        "class_max": int(max(class_counts.values())),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {output_name}: {out_path}")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--source-format", required=True, choices=["snap", "foursquare"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--feature-dim", type=int, default=446)
    parser.add_argument("--num-classes", type=int, default=30)
    parser.add_argument("--min-user-checkins", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    build_location_style_npz(
        raw_path=args.raw_path,
        source_format=args.source_format,
        output_dir=args.output_dir,
        output_name=args.output_name,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        min_user_checkins=args.min_user_checkins,
        max_users=args.max_users,
        seed=args.seed)


if __name__ == "__main__":
    main()
