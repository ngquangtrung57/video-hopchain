
import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq

BASE = "__SCRATCH__/rl/datasets/hopchain_v40_vript_combined"
SRC_PREFIX = "file://__SCRATCH__/tmp/vript/videos/"
DST_PREFIX = "file://__SCRATCH__/videoreason/proxy_vript_140f50k/"


def repoint_cell(cell, kept_source):
    if cell is None:
        return cell, 0, 0
    out, changed, fallback = [], 0, 0
    for elem in cell:
        if isinstance(elem, dict) and isinstance(elem.get("video"), str) and elem["video"].startswith(SRC_PREFIX):
            new_path = DST_PREFIX + elem["video"][len(SRC_PREFIX):]
            if os.path.exists(new_path.replace("file://", "")):
                e = dict(elem)
                e["video"] = new_path
                out.append(e)
                changed += 1
            else:
                out.append(elem)
                fallback += 1
                kept_source.add(elem["video"])
        else:
            out.append(elem)
    return out, changed, fallback


def process(split: str, apply: bool) -> bool:
    src = f"{BASE}/{split}.parquet"
    dst = f"{BASE}/{split}_proxy.parquet"
    table = pq.read_table(src)
    videos = table.column("videos").to_pylist()

    new_videos, total_changed, total_fallback = [], 0, 0
    missing, kept_source = [], set()
    for cell in videos:
        nc, n, fb = repoint_cell(cell, kept_source)
        new_videos.append(nc)
        total_changed += n
        total_fallback += fb
        for elem in nc or []:
            if isinstance(elem, dict) and isinstance(elem.get("video"), str):
                p = elem["video"].replace("file://", "")
                if not os.path.exists(p):
                    missing.append(p)

    idx = table.schema.get_field_index("videos")
    new_col = pa.array(new_videos, type=table.schema.field(idx).type)
    out = table.set_column(idx, table.schema.field(idx), new_col)

    assert out.num_rows == table.num_rows, "row count changed"
    assert out.schema.equals(table.schema), "schema drift"
    for name in table.column_names:
        if name != "videos":
            assert out.column(name).equals(table.column(name)), f"column {name} mutated"

    uniq_missing = sorted(set(missing))
    print(f"{split}: rows={out.num_rows} vript_paths_rewritten={total_changed} "
          f"kept_raw_source={total_fallback} (over {len(kept_source)} distinct videos) "
          f"missing_files={len(uniq_missing)}")
    for p in sorted(kept_source)[:5]:
        print(f"   NO PROXY, kept raw: {p}")
    for p in uniq_missing[:5]:
        print(f"   MISSING {p}")

    if uniq_missing:
        print(f"{split}: REFUSING to write -- {len(uniq_missing)} referenced files do not exist")
        return False

    if apply:
        pq.write_table(out, dst, compression="snappy")
        print(f"{split}: wrote {dst}")
    else:
        print(f"{split}: DRY RUN (rerun with --apply)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    ok = True
    for split in ("train", "val_500"):
        ok &= process(split, args.apply)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
