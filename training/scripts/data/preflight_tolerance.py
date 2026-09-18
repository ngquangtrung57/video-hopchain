#!/usr/bin/env python3

import argparse
import os
import sys

REPO = "__HOME__/training"
DEFAULT_PARQUET = (
    "__SCRATCH__/rl/datasets/"
    "hopchain/hopchain_val.parquet"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--n", type=int, default=5, help="rows to test")
    args = ap.parse_args()

    if REPO not in sys.path:
        sys.path.insert(0, REPO)

    import pandas as pd

    try:
        from rewards.boxed_reward_wrapper import compute_score
    except Exception as e:
        print(f"FAIL: cannot import compute_score: {type(e).__name__}: {e}")
        return 1
    print("ok: rewards.boxed_reward_wrapper imported")

    df = pd.read_parquet(args.parquet)

    def gold_of(row):
        return float(row["reward_model"]["ground_truth"])

    big = df[df.apply(lambda r: gold_of(r) >= 150, axis=1)]
    if big.empty:
        print(f"FAIL: no rows with gold >= 150 in {args.parquet}; test cannot discriminate")
        return 1

    prefill = os.environ.get("VERL_THINK_PREFILL", "0") == "1"
    print(f"ok: {len(big)} candidate rows (gold >= 150); VERL_THINK_PREFILL={int(prefill)}")

    failures = []
    for i in range(min(args.n, len(big))):
        row = big.iloc[i]
        gt = row["reward_model"]["ground_truth"]
        extra = dict(row["extra_info"])

        if "tolerance" not in extra:
            failures.append(f"row {i}: extra_info has no 'tolerance' key")
            continue

        think = (
            "Checking each hop in turn against the separated moments in the video, "
            "then adding the per-hop values to obtain the total for this chain."
        )
        gold_resp = f"<think>{think}</think>\\boxed{{{gt}}}"
        near_resp = f"<think>{think}</think>\\boxed{{{float(gt) + 1:g}}}"

        g = compute_score("", gold_resp, gt, extra_info=extra)
        w = compute_score("", near_resp, gt, extra_info=extra)

        gs = g["accuracy"] if isinstance(g, dict) else g
        ws = w["accuracy"] if isinstance(w, dict) else w

        ok = gs > 0.9 and ws < 0.5
        print(
            f"  row {i}: tolerance={extra['tolerance']} gold={gt} "
            f"gold_acc={gs:.3f} gold+1_acc={ws:.3f} {'OK' if ok else 'BAD'}"
        )
        if not ok:
            if ws >= 0.5:
                failures.append(
                    f"row {i}: gold+1 scored accuracy {ws:.3f} -- the 1% fallback band is ACTIVE, "
                    "extra_info.tolerance is not reaching the scorer"
                )
            else:
                failures.append(f"row {i}: gold itself scored accuracy only {gs:.3f}")

    if failures:
        print("\nFAIL: reward signal is NOT intact")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nPASS: tolerance=0.0 reaches compute_score; gold+1 is correctly rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
