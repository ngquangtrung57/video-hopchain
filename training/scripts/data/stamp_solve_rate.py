#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def band(sr: float) -> str:
    if sr <= 0.0:
        return "dead_hard"
    if sr >= 1.0:
        return "dead_easy"
    if sr <= 0.25:
        return "hard"
    if sr <= 0.75:
        return "medium"
    return "easy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True, help="dir of probe_*.jsonl sidecars")
    ap.add_argument("--parquet", required=True, action="append",
                    help="corpus parquet to stamp. Repeatable")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suffix", default="_probed")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.probe_dir, "*.jsonl")))
    if not files:
        print(f"ERROR: no *.jsonl under {a.probe_dir}", file=sys.stderr)
        return 1
    probe: dict[str, list[int]] = {}
    n_lines = dupes = errs = 0
    for path in files:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                n_lines += 1
                qid = d.get("question_id")
                if qid is None or d.get("error"):
                    errs += 1
                    continue
                e = probe.get(qid)
                if e is None:
                    probe[qid] = [int(d["n_samples"]), int(d["n_correct"])]
                else:
                    dupes += 1
                    e[0] += int(d["n_samples"])
                    e[1] += int(d["n_correct"])
    print(f"probe: {len(files)} sidecars, {n_lines} rows, {len(probe)} distinct question_id, "
          f"{dupes} duplicate rows merged, {errs} errored rows skipped")
    if probe:
        ns = collections.Counter(v[0] for v in probe.values())
        print(f"  samples-per-question: {dict(sorted(ns.items()))}")

    os.makedirs(a.out_dir, exist_ok=True)
    for src in a.parquet:
        table = pq.read_table(src)
        rows = table.to_pylist()
        hit = 0
        for r in rows:
            ei = r["extra_info"]
            e = probe.get(ei.get("question_id"))
            if e is None:
                continue
            n, c = e
            if n <= 0:
                continue
            sr = c / n
            ei["solve_rate"] = float(sr)
            ei["difficulty_band"] = band(sr)
            hit += 1
        name = os.path.basename(src).replace(".parquet", f"{a.suffix}.parquet")
        dest = os.path.join(a.out_dir, name)

        srs = [r["extra_info"]["solve_rate"] for r in rows
               if r["extra_info"]["solve_rate"] >= 0]
        live = sum(1 for s in srs if 0 < s < 1)
        bands = collections.Counter(r["extra_info"]["difficulty_band"] for r in rows)
        print(f"\n{os.path.basename(src)}: {len(rows)} rows, stamped {hit} "
              f"({hit/len(rows):.1%} coverage)")
        if srs:
            print(f"  measured {len(srs)} | mean solve_rate {sum(srs)/len(srs):.4f} | "
                  f"LIVE (0<sr<1) {live} ({live/len(srs):.2%} of measured)")
            hist = collections.Counter(round(s, 4) for s in srs)
            print("  solve_rate histogram:",
                  {k: v for k, v in sorted(hist.items())[:12]})
        print(f"  bands: {dict(bands.most_common())}")

        assert len(rows) == table.num_rows, "row count changed"
        if a.dry_run:
            print(f"  [dry-run] would write {dest}")
            continue
        out = pa.Table.from_pylist(rows, schema=table.schema)
        assert out.num_rows == table.num_rows
        assert out.schema.equals(table.schema), "schema drifted"
        pq.write_table(out, dest)
        print(f"  wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
