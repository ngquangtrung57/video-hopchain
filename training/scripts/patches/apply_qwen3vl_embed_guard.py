#!/usr/bin/env python3
import sys

try:
    import vllm.model_executor.models.utils as _u
except Exception as e:
    print(f"[apply_mm_embed_guard] could not import vllm utils ({e}); skipping",
          file=sys.stderr)
    sys.exit(0)

path = _u.__file__
src = open(path).read()

if "[mm-embed-guard]" in src:
    print(f"[apply_mm_embed_guard] already patched: {path}")
    sys.exit(0)

OLD = '''    flattened = _flatten_embeddings(multimodal_embeddings)
    try:'''

NEW = '''    flattened = _flatten_embeddings(multimodal_embeddings)
    # [mm-embed-guard] When there are more multimodal placeholder positions than
    # embedding rows, cap the excess (last) True positions to the number of rows
    # so masked_scatter_ stays in bounds; the extra placeholders keep their text
    # embedding.
    _mm_num_ph = int(is_multimodal.sum().item())
    _mm_src = int(flattened.shape[0])
    if _mm_num_ph > _mm_src:
        print(f"[mm-embed-guard] multimodal placeholders {_mm_num_ph} > "
              f"embedding rows {_mm_src}; capping excess to text embeddings "
              f"(no crash)", flush=True)
        _mm_true = is_multimodal.nonzero(as_tuple=False).flatten()
        is_multimodal = is_multimodal.clone()
        is_multimodal[_mm_true[_mm_src:]] = False
    try:'''

n = src.count(OLD)
if n != 1:
    print(
        f"[apply_mm_embed_guard] expected exactly 1 match of the anchor, found {n}; "
        f"NOT patching {path} (vLLM version changed? re-derive the patch manually)",
        file=sys.stderr,
    )
    sys.exit(1)

open(path, "w").write(src.replace(OLD, NEW))
print(f"[apply_mm_embed_guard] patched OK (cap): {path}")
