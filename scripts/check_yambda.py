"""Data sufficiency checks for Yambda-50M (listens).

Answers, before any training:
  1. What share of items (and of events) has an audio embedding, i.e. can get an audio-based SID?
  2. Under a global temporal split, how many test targets are re-listens vs never-seen items?
  3. How large are the groups for conditional coverage (user activity, item popularity, seen/unseen)?
  4. How many new items appear per week (drift)?
"""
import json
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "yambda50m"
OUT = ROOT / "data" / "processed"
EMB_URL = "https://huggingface.co/datasets/yandex/yambda/resolve/main/embeddings.parquet"
DAY = 24 * 3600  # timestamps are seconds, rounded to multiples of 5 (checked: span = 301 days)


def embedding_item_ids():
    """Read only the item_id column of the 13.8 GB embeddings file over HTTP range requests."""
    cache = OUT / "yambda_embedding_item_ids.npy"
    if cache.exists():
        return np.load(cache)
    with fsspec.open(EMB_URL, "rb", block_size=2**22) as f:
        pf = pq.ParquetFile(f)
        print(f"  embeddings.parquet: {pf.metadata.num_rows:,} rows, {pf.num_row_groups} row groups")
        ids = np.concatenate([pf.read_row_group(i, columns=["item_id"]).column(0).to_numpy()
                              for i in range(pf.num_row_groups)])
    np.save(cache, ids)
    return ids


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW / "listens.parquet", columns=["uid", "item_id", "timestamp", "played_ratio_pct"])
    days = (df.timestamp.max() - df.timestamp.min()) / DAY
    print(f"listens {len(df):,}  users {df.uid.nunique():,}  items {df.item_id.nunique():,}  span {days:.0f} days")

    # 1. Audio-embedding coverage
    emb_ids = embedding_item_ids()
    has_emb = df.item_id.isin(set(emb_ids.tolist()))
    items = df.item_id.unique()
    item_cov = np.isin(items, emb_ids).mean()
    print(f"\n[embeddings] {len(emb_ids):,} tracks with audio; coverage of 50M items {item_cov:.1%}, of events {has_emb.mean():.1%}")

    # 2. Global temporal split: last 7 days = test, previous 7 days = calibration, rest = train
    t_end = df.timestamp.max()
    t_test, t_cal = t_end - 7 * DAY, t_end - 14 * DAY
    train = df[df.timestamp <= t_cal]
    cal = df[(df.timestamp > t_cal) & (df.timestamp <= t_test)]
    test = df[df.timestamp > t_test]
    seen_items = set(train.item_id)
    res = {"listens": len(df), "items": int(len(items)), "span_days": float(days),
           "item_embedding_coverage": float(item_cov), "event_embedding_coverage": float(has_emb.mean())}
    for name, part in [("calibration", cal), ("test", test)]:
        first = part.groupby("uid").head(1)  # next item after the history cut, one per user
        first = first[first.uid.isin(set(train.uid))]
        unseen = ~first.item_id.isin(seen_items)
        user_seen = first.merge(train[["uid", "item_id"]].drop_duplicates(), on=["uid", "item_id"], how="left", indicator=True)
        relisten = (user_seen["_merge"] == "both").mean()
        emb_unseen = first.loc[unseen, "item_id"].isin(set(emb_ids.tolist())).mean() if unseen.any() else float("nan")
        all_unseen = ~part.item_id.isin(seen_items)
        print(f"\n[{name}] users {len(first):,}; first-target re-listen by same user {relisten:.1%}; "
              f"item never in train {unseen.mean():.1%} ({int(unseen.sum()):,} users); "
              f"unseen targets with audio {emb_unseen:.1%}")
        print(f"  all {name} events {len(part):,}; events on never-in-train items {all_unseen.mean():.1%} "
              f"({part.loc[all_unseen, 'item_id'].nunique():,} distinct new items)")
        res[name] = {"users": int(len(first)), "relisten_share": float(relisten),
                     "unseen_item_share": float(unseen.mean()), "unseen_with_audio": float(emb_unseen)}

    # 3. Group sizes: user history length and target popularity (test first targets)
    first = test.groupby("uid").head(1)
    first = first[first.uid.isin(set(train.uid))]
    hist = train.groupby("uid").size().reindex(first.uid).to_numpy()
    print(f"\n[user activity] train history quantiles 10/25/50/75/90%: {np.quantile(hist, [.1, .25, .5, .75, .9]).astype(int).tolist()}")
    pop = train.item_id.value_counts()
    tp = first.item_id.map(pop).fillna(0).to_numpy()
    for label, lo, hi in [("unseen (0)", 0, 0), ("tail (1-10)", 1, 10), ("mid (11-100)", 11, 100), ("head (>100)", 101, 10**12)]:
        print(f"  target popularity {label}: {int(((tp >= lo) & (tp <= hi)).sum()):,} test users")

    # 4. Drift: new items per week
    week = ((df.timestamp - df.timestamp.min()) // (7 * DAY)).astype(int)
    first_seen = df.groupby("item_id").timestamp.min()
    fs_week = ((first_seen - df.timestamp.min()) // (7 * DAY)).astype(int)
    per_week = pd.DataFrame({"events": week.value_counts().sort_index(), "new_items": fs_week.value_counts().sort_index()}).fillna(0).astype(int)
    print("\n[drift] events and first-appearing items per week (first/last 3 weeks):")
    print(pd.concat([per_week.head(3), per_week.tail(3)]).to_string())
    res["new_items_last_4_weeks_mean"] = float(per_week.new_items.tail(4).mean())
    (OUT / "yambda50m_checks.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
