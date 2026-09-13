#!/usr/bin/env python3
import sys

try:
    import vllm.model_executor.layers.rotary_embedding.mrope as _m
except Exception as e:
    print(f"[apply_mrope_guard] could not import vllm mrope ({e}); skipping", file=sys.stderr)
    sys.exit(0)

path = _m.__file__
src = open(path).read()

if "[mrope-guard]" in src:
    print(f"[apply_mrope_guard] already patched: {path}")
    sys.exit(0)

OLD = '''        video_grid_thw = [[1, h, w] for t, h, w in video_grid_thw
                          for _ in range(t)]

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums'''

NEW = '''        video_grid_thw = [[1, h, w] for t, h, w in video_grid_thw
                          for _ in range(t)]

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        # [mrope-guard] Cap the image and video segment counts to the number of
        # grid entries available, so the position loop cannot over-index the grid
        # or look past the last segment. Extra video tokens fall through to the
        # trailing text-position block.
        if int(video_nums) > len(video_grid_thw):
            print(f"[mrope-guard] video segments {int(video_nums)} > grid frames "
                  f"{len(video_grid_thw)}; capping (extra video tokens get text "
                  f"positions, no crash)", flush=True)
            video_nums = len(video_grid_thw)
        if int(image_nums) > len(image_grid_thw):
            image_nums = len(image_grid_thw)
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums'''

n = src.count(OLD)
if n != 1:
    print(
        f"[apply_mrope_guard] expected exactly 1 match of the target block, found {n}; "
        f"NOT patching {path} (vLLM version changed? re-derive the patch manually)",
        file=sys.stderr,
    )
    sys.exit(1)

open(path, "w").write(src.replace(OLD, NEW))
print(f"[apply_mrope_guard] patched OK (cap): {path}")
