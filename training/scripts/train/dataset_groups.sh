#!/bin/bash
# Parquet groups consumed by the training launchers, and the helper that turns a
# group into the bracketed list the trainer expects.

HOPCHAIN_DIR="${HOPCHAIN_DIR:-${SCRATCH_DIR}/datasets/hopchain}"

join_to_list() {
    local IFS=','
    echo "[$*]"
}

# The published dataset ships videohopchain_{train,val}.parquet. Older local runs
# used hopchain_{train,val}.parquet, so accept either and fail loudly on neither,
# because verl reads a missing parquet as an empty dataset and trains on nothing.
hopchain_split() {
    local split="$1" f
    for f in "${HOPCHAIN_DIR}/videohopchain_${split}.parquet" \
             "${HOPCHAIN_DIR}/hopchain_${split}.parquet"; do
        if [[ -f "$f" ]]; then echo "$f"; return 0; fi
    done
    echo "dataset_groups.sh: no ${split} parquet under ${HOPCHAIN_DIR}." >&2
    echo "  Download it with:" >&2
    echo "  huggingface-cli download ngqtrung/video-hopchain --repo-type dataset \\" >&2
    echo "      --local-dir \"${HOPCHAIN_DIR}\"" >&2
    return 1
}

GROUP_VIDEO_TRAIN_HOPCHAIN=(
    "$(hopchain_split train)"
)
GROUP_VAL_HOPCHAIN=(
    "$(hopchain_split val)"
)
