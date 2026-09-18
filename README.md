# Video-HopChain: Multi-Hop Questions and Confidence-Gated Exploration for Video Reasoning Models

[**Paper**](https://arxiv.org/abs/XXXX.XXXXX) &nbsp;|&nbsp;
[**Dataset**](https://huggingface.co/datasets/ngqtrung/video-hopchain) &nbsp;|&nbsp;
[**Model**](https://huggingface.co/ngqtrung/video-hopchain-8b) &nbsp;|&nbsp;
[**Collection**](https://huggingface.co/collections/ngqtrung/video-hopchain)

Code for the paper. `pipeline/` builds the Video-HopChain corpus from raw videos. `training/`
is a fork of [verl](https://github.com/volcengine/verl) that adds Confidence-Gated Exploration to the
fully asynchronous GRPO trainer. The paper specifies the method; this file says where things
live and how to run them.

## What this is

**Video-HopChain** is a dataset of multi-hop video questions. Every question chains three to six
yes/no hops about moments in one video, each hop selects one of two integers, and the answer is
their sum. The integers are resampled until every combination of hop answers gives a different
sum, so a wrong intermediate hop changes the total. One exact match on one number therefore gives
the verifiable reward that RLVR needs.

**Confidence-Gated Exploration (CGE)** recovers the groups that GRPO throws away. With 8 rollouts per
question, CGE draws the first 4 normally. If all 4 earn the same accuracy, the group carries no
reward variance and no gradient, so CGE draws the last 4 with the policy's top token masked
wherever its probability exceeds 0.95, inside the reasoning span only. The masked positions leave
the loss while all 8 rollouts enter the group advantage, so the method spends no extra rollouts.

Accuracy in percent, `lmms-eval` at 100 frames per video, one setting for every row.

| model | Video-MME | Perception-Comp | Video-MMMU | Video-Holmes | VCRBench | MMR-V | LongVideo-Reason | VRBench | mean |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | 64.0 | 28.1 | 63.5 | 40.7 | 32.0 | 42.7 | 72.6 | 74.7 | 52.3 |
| + standard RL | 65.8 | 34.3 | 63.0 | 47.4 | 35.9 | 43.4 | 76.1 | 77.6 | 55.4 |
| + standard RL + Video-HopChain | 68.6 | 36.2 | 64.7 | **48.6** | 42.1 | 45.9 | 77.8 | 79.4 | 57.9 |
| + standard RL + Video-HopChain + CGE | **69.2** | **37.4** | **67.5** | **48.6** | **44.9** | **46.6** | **78.9** | **81.0** | **59.3** |

## Get the data and the model

```bash
hf download ngqtrung/video-hopchain --repo-type dataset --local-dir video-hopchain
cd video-hopchain && for z in videohopchain_videos_*.zip; do unzip -q "$z"; done
```

That gives `videohopchain_{train,val}.parquet` (22,550 and 1,000 rows) and `videos/<source>/<id>.mp4`,
which is the path the parquet rows already point at. The trained checkpoint is
[`ngqtrung/video-hopchain-8b`](https://huggingface.co/ngqtrung/video-hopchain-8b).

You only need `pipeline/` if you want to build a corpus of your own.

## Environments

The two halves pin different versions and **cannot share one environment**. Build two.

| | Python | key pins |
|---|---|---|
| `pipeline/` | 3.12 | `pyarrow==24.0.0`, `opencv-python==5.0.0.93`, `scenedetect==0.7`; extras `[serve]`, `[caption-gemini]`, `[fetch]` |
| `training/` | 3.12 | `transformers==4.57.0`, `numpy==1.26.4`, `torch==2.8.0`, `vllm~=0.13.0` |

Install `pipeline` from its `pyproject.toml`, not from `requirements.txt`, which is unpinned and
does not reproduce our runs.

```bash
pip install -e pipeline            # add [serve] for a local vLLM captioner
pip install -e training/verl       # the training half, in its own environment
```

## Layout

```
pipeline/
  pipeline/          one module per stage, plus the four prompts, the model client and the
                     hop record
  run.py             runs the stages in order over a folder of videos
training/
  verl/              the verl fork that implements Confidence-Gated Exploration
  rewards/           the reward function
  scripts/patches/   two guards applied to an installed vLLM for Qwen3-VL video inputs
  scripts/train/     the shared training environment and the launch scripts
```

## The pipeline

| stage | module | what it does |
|---|---|---|
| 1 admit and encode | `pipeline/proxy.py` | keep videos of at least 180 s, and write the frame proxy |
| 2 segment | `pipeline/segment.py` | split the video into shots |
| 3 caption | `pipeline/caption.py` | caption every shot with a vision-language model |
| 4 chain specification | `pipeline/spec.py` | draw hop count, families, branch numbers and links |
| 5 generate | `pipeline/generate.py` | a text-only model fills each specification from the captions |
| 6 code checks | `pipeline/checks.py` | recompute every answer, and drop a candidate whose arithmetic fails |
| 7 judge | `pipeline/judge.py` | a text-only model faults each hop against the same captions |
| 8 regenerate | `pipeline/generate.py` | a faulted question goes back to the generator and to the judge, until it passes |
| 9 difficulty filter | `pipeline/difficulty.py` | drop what the base model already solves |
| 10 assemble | `pipeline/assemble.py` | render the rows, and split by video |

The four prompts are in `pipeline/prompts.py`: `P1_CAPTION` for the captioner, `P_HOPGEN` for
the generator, `P_JUDGE` for the judge and `P_SOLVER` for the difficulty filter.

Every model stage calls an OpenAI-compatible endpoint through `pipeline/model.py`, so any
server that speaks that API will do, a local vLLM server included.

```bash
cd pipeline
python run.py --videos VIDEOS --out WORK \
    --caption-model CAPTIONER --text-model GENERATOR --solver-model BASE_MODEL \
    --base-url http://localhost:8000/v1 --held-out 100
```

Each stage writes its artefact under `--out`: the proxies, one caption document per video, the
surviving questions per video, and the train and validation parquet files. Rerunning skips a
video whose caption document is already there.

## The training

`training/scripts/train/dataset_groups.sh` maps a dataset group to the parquet files and
`training/scripts/train/common_env.sh` holds the environment both launchers share. Then launch
one arm:

| arm | script |
|---|---|
| plain GRPO | `training/scripts/train/hopchain/launch_plain.sh` |
| Confidence-Gated Exploration | `training/scripts/train/hopchain/launch_cge.sh` |

Both call `training/scripts/train/hopchain/run_4node_hopchain.sh`, which sets the environment
and then calls `training/scripts/train/hopchain/grpo_video_4node_8b_hopchain.sh` with every
hyperparameter. `training/scripts/launch_pbs_4node_separate_jobs.sh` starts the Ray cluster
when the nodes come from separate scheduler jobs.

```bash
pip install -e training/verl
python training/scripts/patches/apply_mrope_video_guard.py
python training/scripts/patches/apply_qwen3vl_embed_guard.py
```

Confidence-Gated Exploration lives in these files of the fork. The implementation keeps the earlier
name of the method in its configuration key `two_wave_enable`, in the metric prefix
`rvrl/two_wave/` and in the test file names.

| part | file |
|---|---|
| the top-token mask, a vLLM logits processor | `training/verl/verl/workers/rollout/exploration/entropy_dropout_lp.py` |
| the configuration | `training/verl/verl/workers/config/rollout.py`, `training/verl/verl/trainer/config/rollout/rollout.yaml` |
| the second-wave sampling and the per-sequence drop mask | `training/verl/verl/experimental/agent_loop/agent_loop.py` |
| the gate, the group statistics and the metrics | `training/verl/verl/experimental/fully_async_policy/fully_async_trainer.py` |
| the loss mask on the masked positions | `training/verl/verl/workers/utils/losses.py` |
| the tests | `training/verl/tests/experimental/agent_loop/test_two_wave_exploration_on_cpu.py` |

The reward function is `training/rewards/vero_reward_wrapper.py`.

The shell scripts read their settings from the environment and abort with a message naming the
variable when one is unset. Set at least `SCRATCH_DIR`, `HOPCHAIN_DIR`, `MODEL_PATH_OVERRIDE`,
`HEAD_NODE`/`HEAD_JOB` and `WORKER{1,2,3}_NODE`/`_JOB`.

### The CGE settings

`launch_cge.sh` differs from `launch_plain.sh` in these variables only, and the values below are
the ones the paper reports.

| variable | plain | CGE | verl config key |
|---|---|---|---|
| `EXPLORE_ENABLE` | `false` | `true` | `rollout.exploration.enable` |
| `CGE_ENABLE` | `false` | `true` | `rollout.exploration.two_wave_enable` |
| `TRIGGER_MODE` | — | `high` | `rollout.exploration.trigger_mode` |
| `TOP_PROB_THRESHOLD` | — | `0.95` | `rollout.exploration.top_prob_threshold` |
| `EXPLORE_MAX_MEAN` | — | `1.0` | `rollout.exploration.explore_max_mean` |

The launcher also fixes `anchor_fraction=0.5` (the first wave is half the group),
`variance_metric=accuracy` (the check reads accuracy, not the composite reward),
`drop_top_k=1`, `mask_from_loss=true` and `restrict_to_think_region=true`. The class defaults in
`workers/config/rollout.py` are weaker than these, so read the launcher, not the defaults.

### Running outside PBS

`run_4node_hopchain.sh` and `launch_pbs_4node_separate_jobs.sh` are reference scripts shaped for
our PBS cluster. They set `PBS_JOBID` before each `ssh`, assume passwordless ssh between allocated
nodes and exactly 8 GPUs per node, and refuse to start when `nvidia-smi` reports any other process
on a GPU. None of that survives Slurm or a single machine. If that is your case, ignore the
launchers and call the underlying trainer directly: the full command line, with every
hyperparameter, is in `training/scripts/train/hopchain/grpo_video_4node_8b_hopchain.sh`.

The two vLLM patches under `training/scripts/patches/` edit an installed vLLM in place, so any
later `pip install vllm` silently removes them. Both exit 0 when vLLM is absent, which means a
failed patch is invisible. Re-run them after every vLLM install.

## Citation

```bibtex
@article{videohopchain2026,
  title   = {Video-HopChain: Multi-Hop Questions and Confidence-Gated Exploration for Video Reasoning Models},
  author  = {Nguyen, Quang Trung and Dong, Yuhao and Sun, Shuo and Liu, Shuai and Tian, Shulin and Yap, Kim-Hui and Liu, Ziwei},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Apache License 2.0, the license of verl, which this release extends. See `LICENSE`.
