#!/usr/bin/env python3
import argparse
import collections
import itertools
import os
import sys

import pyarrow.parquet as pq

SRC = ("__SCRATCH__/rl/datasets/"
       "mix63k_ordered/mix63k_ordered_train.parquet")
OUT = ("__SCRATCH__/rl/datasets/"
       "multihop30k/multihop30k_train.parquet")

BOUNDARY = 33462
EXPECTED_ROWS = 30000
PHASE_A_PREFIX = "video_r1_"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    table = pq.read_table(args.src)
    n = table.num_rows
    src_col = table.column("data_source").to_pylist()

    if src_col[BOUNDARY - 1].startswith(PHASE_A_PREFIX) is False:
        print(f"ABORT: row {BOUNDARY-1} is {src_col[BOUNDARY-1]!r}, expected a "
              f"{PHASE_A_PREFIX}* row", file=sys.stderr)
        return 2
    if src_col[BOUNDARY].startswith(PHASE_A_PREFIX):
        print(f"ABORT: row {BOUNDARY} is {src_col[BOUNDARY]!r}, still phase A",
              file=sys.stderr)
        return 2

    tail = table.slice(BOUNDARY, n - BOUNDARY)
    tail_src = src_col[BOUNDARY:]

    leaked = [s for s in tail_src if s.startswith(PHASE_A_PREFIX)]
    if leaked:
        print(f"ABORT: {len(leaked)} phase-A rows in the tail", file=sys.stderr)
        return 2
    if tail.num_rows != EXPECTED_ROWS:
        print(f"ABORT: tail has {tail.num_rows} rows, expected {EXPECTED_ROWS}",
              file=sys.stderr)
        return 2

    runs = [len(list(g)) for _, g in itertools.groupby(tail_src)]
    print(f"rows        {tail.num_rows}")
    print(f"sources     {dict(collections.Counter(tail_src))}")
    print(f"interleave  {len(runs)} runs, longest {max(runs)}")
    print(f"per epoch   {tail.num_rows / 512:.1f} param_versions "
          f"(512 prompts/pv = 128 x trigger_parameter_sync_step 4)")

    if args.check_only:
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pq.write_table(tail, args.out, compression="snappy")
    back = pq.ParquetFile(args.out)
    if back.metadata.num_rows != EXPECTED_ROWS:
        print(f"ABORT: wrote {back.metadata.num_rows} rows", file=sys.stderr)
        return 2
    print(f"wrote       {args.out} "
          f"({os.path.getsize(args.out)/1e6:.1f} MB, {back.metadata.num_rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
