
import argparse
import collections
import os

import numpy as np
import pyarrow.parquet as pq

BASE = "__SCRATCH__/rl/datasets/hopchain_v40_vript_combined"
SEED = 0


def video_path_of(row):
    for v in row:
        if isinstance(v, dict) and "video" in v:
            return str(v["video"])
    return None


def video_of(row):
    p = video_path_of(row)
    return os.path.basename(p) if p else None


def source_of(vid_path):
    return "vript" if "vript" in vid_path.lower() else "hopchain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=f"{BASE}/val_500.parquet")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    val = pq.read_table(f"{BASE}/val.parquet")
    train_vids = {
        video_of(r)
        for r in pq.read_table(f"{BASE}/train.parquet", columns=["videos"]).column("videos").to_pylist()
    }

    rows = val.column("videos").to_pylist()
    vids = [video_of(r) for r in rows]
    src_by_vid = {video_of(r): source_of(video_path_of(r)) for r in rows}
    by_video = collections.defaultdict(list)
    for i, v in enumerate(vids):
        by_video[v].append(i)

    leak = set(by_video) & train_vids
    assert not leak, f"val videos also in train: {sorted(leak)[:5]}"

    src_rows = collections.Counter(src_by_vid[v] for v in vids)
    total = sum(src_rows.values())
    targets = {s: round(args.n * n / total) for s, n in src_rows.items()}

    rng = np.random.default_rng(SEED)
    keep = []
    for src, target in sorted(targets.items()):
        pool = sorted(v for v in by_video if src_by_vid[v] == src)
        rng.shuffle(pool)
        taken = 0
        for v in pool:
            if taken >= target:
                break
            idxs = by_video[v][: target - taken]
            keep.extend(idxs)
            taken += len(idxs)
        print(f"  {src}: target={target} got={taken}")

    keep = sorted(keep)[: args.n]
    out = val.take(keep)
    assert out.schema.equals(val.schema), "schema drift"

    kept_vids = {video_of(r) for r in out.column("videos").to_pylist()}
    print(f"rows={out.num_rows} videos={len(kept_vids)} "
          f"train_overlap={len(kept_vids & train_vids)}")
    print("  by source:", dict(collections.Counter(src_by_vid[v] for v in kept_vids)))

    if args.apply:
        pq.write_table(out, args.out, compression="snappy")
        print(f"wrote {args.out}")
    else:
        print("DRY RUN — rerun with --apply")


if __name__ == "__main__":
    main()
