#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="PHASE_NAME:PARQUET, in the order they should train. Repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-inner-shuffle", action="store_true",
                    help="keep each input's row order verbatim instead of seed-shuffling it")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows, schema, phases = [], None, []
    for spec in a.input:
        name, path = spec.split(":", 1)
        table = pq.read_table(path)
        if schema is None:
            schema = table.schema
        elif not table.schema.equals(schema):
            print(f"ERROR: {path} schema differs from the first input. verl reads the parquets as "
                  f"one dataset, so the schemas must match field for field.", file=sys.stderr)
            return 1
        chunk = table.to_pylist()
        if not a.no_inner_shuffle:
            rng.shuffle(chunk)
        start = len(rows)
        rows.extend(chunk)
        phases.append((name, start, len(rows), path))
        print(f"phase {len(phases)}: {name:12s} rows {start:6d}..{len(rows):6d} "
              f"({len(chunk)} rows) <- {os.path.basename(path)}")

    print(f"\ntotal {len(rows)} rows")
    for name, s0, s1, _ in phases:
        seg = rows[s0:s1]
        srs = [r["extra_info"]["solve_rate"] for r in seg if r["extra_info"]["solve_rate"] >= 0]
        live = sum(1 for x in srs if 0 < x < 1)
        ds = collections.Counter(r["data_source"] for r in seg)
        rt = collections.Counter(r["extra_info"]["reward_type"] for r in seg)
        print(f"  {name}: rows {len(seg)} | reward_type {dict(rt)}")
        if srs:
            print(f"      measured {len(srs)} | mean solve_rate {sum(srs)/len(srs):.4f} | "
                  f"live {live} ({live/len(srs):.2%})")
        print(f"      sources {dict(ds.most_common())}")
        v = seg[0]["videos"][0]
        print(f"      decode: {v['max_frames']}f @ {v['max_pixels']}px")

    order = [r["data_source"] for r in rows]
    seen, blocks = set(), 0
    prev = None
    for d in order:
        if d != prev:
            blocks += 1
            prev = d
    per_phase_sources = [set(r["data_source"] for r in rows[s0:s1]) for _, s0, s1, _ in phases]
    for i, a_set in enumerate(per_phase_sources):
        for j, b_set in enumerate(per_phase_sources):
            if i < j:
                assert not (a_set & b_set), \
                    f"phases {phases[i][0]} and {phases[j][0]} share a data_source; the boundary " \
                    f"would be unverifiable"
    qids = [r["extra_info"]["question_id"] for r in rows]
    assert len(qids) == len(set(qids)), "duplicate question_id across the inputs"
    print(f"\nphase boundaries verified: no data_source appears in two phases; "
          f"{len(set(qids))} unique question_id")

    if a.dry_run:
        print("[dry-run] nothing written")
        return 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out = pa.Table.from_pylist(rows, schema=schema)
    assert out.num_rows == len(rows)
    pq.write_table(out, a.out)
    print(f"wrote {a.out} ({out.num_rows} rows)")
    print("\nREMINDER: set data.shuffle=false in the launcher, or this ordering is discarded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
