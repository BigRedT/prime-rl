# W&B Metrics Reference

All metrics logged to W&B during RL training, grouped by prefix.

---

## Inference (`inference/`)

Collected from vLLM Prometheus endpoints, smoothed over a 20-step window.

| Metric | Description |
|---|---|
| `inference/num_requests_running` | Requests currently being processed |
| `inference/num_requests_waiting` | Requests queued |
| `inference/gpu_cache_usage_perc_max` | Max KV cache usage across engines |
| `inference/gpu_cache_usage_perc_mean` | Mean KV cache usage |
| `inference/gpu_prefix_cache_hit_rate_max` | Max prefix cache hit rate |
| `inference/gpu_prefix_cache_hit_rate_mean` | Mean prefix cache hit rate |
| `inference/prefill_throughput_tps` | Prefill tokens/sec |
| `inference/decode_throughput_tps` | Decode tokens/sec |
| `inference/completed_requests_per_s` | Request completions/sec |
| `inference/nixl_xfer_time_seconds_avg_ms` | Avg NIXL weight transfer time (ms) |

---

## Progress (`progress/`)

| Metric | Description |
|---|---|
| `progress/tokens` | Tokens generated this step |
| `progress/prefill_tokens` | Prefill tokens this step |
| `progress/decode_tokens` | Decode tokens this step |
| `progress/samples` | Rollouts generated this step |
| `progress/problems` | Unique prompts in this batch |
| `progress/total_tokens` | Cumulative tokens |
| `progress/total_samples` | Cumulative rollouts |
| `progress/total_problems` | Cumulative unique prompts |
| `progress/ckpt_step` | Current training step |

---

## Performance (`perf/`)

| Metric | Description |
|---|---|
| `perf/throughput` | Tokens trained on per second (forward/backward pass) across all GPUs |
| `perf/throughput_per_gpu` | Tokens trained on per second per GPU |
| `perf/mfu` | Model FLOPs Utilization (%) |
| `perf/peak_memory` | Peak GPU memory reserved (GiB) |

---

## Loss & Optimizer

| Metric | Description |
|---|---|
| `loss/mean` | Mean training loss |
| `entropy/mean` | Mean policy entropy |
| `mismatch_kl/mean` | KL divergence between policy and inference logprobs |
| `kl_ent_ratio/mean` | `mismatch_kl/mean ÷ entropy/mean` |
| `optim/lr` | Learning rate |
| `optim/grad_norm` | Gradient norm (if clipping enabled) |
| `optim/zero_grad_ratio` | Fraction of zero gradients |

---

## Timing (`time/`)

| Metric | Description |
|---|---|
| `time/step` | Total step wall time |
| `time/wait_for_batch` | Waiting for rollout batch to arrive |
| `time/load_data` | Loading batch onto GPU |
| `time/broadcast_weights` | Weight broadcast to inference engine |
| `time/forward_backward` | Forward + backward pass |
| `time/save_ckpt` | Checkpoint save time |
| `time/generate_completions` | Rollout generation time (orchestrator) |

---

## Reward & Solve Rates

Per-environment (replace `{env}` with the env name) and `all` (aggregated).

| Metric | Description |
|---|---|
| `reward/{env}/mean` | Mean reward |
| `solve_none/{env}` | Fraction of examples with zero reward on all rollouts |
| `solve_all/{env}` | Fraction of examples with perfect reward on all rollouts |
| `effective_batch_size/{env}` | Fraction that are neither `solve_none` nor `solve_all` |
| `batch/{env}` | Fraction of rollouts from this env |
| `metrics/{env}/{metric}` | Custom metrics from the environment's reward function |

---

## Sequence Length & Generation Stats

Per-environment and `all`. Each has `/mean`, `/max`, `/min` unless noted.

| Metric | Description |
|---|---|
| `seq_len/{env}/mean` | Mean total sequence length |
| `decode_len/{env}/mean` | Mean decode (completion) length |
| `prefill_len/{env}/mean` | Mean prefill (prompt) length |
| `is_truncated/{env}/mean` | Fraction of truncated generations (no `/min`) |
| `num_turns/{env}/mean` | Mean number of turns (multi-turn envs) |
| `generation_ms/{env}/mean` | Mean generation latency (ms) |
| `scoring_ms/{env}/mean` | Mean scoring latency (ms) |
| `samples_per_rollout/{env}/mean` | Mean training samples extracted per rollout |
| `stop_condition/{env}/{reason}` | Fraction of rollouts ending with each stop reason |

---

## Evaluation (`eval/`)

Logged when eval is triggered. Replace `{env}` with the eval env name, `{k}` with rollouts-per-example.

| Metric | Description |
|---|---|
| `eval/{env}/avg@{k}` | Average reward over k rollouts per example |
| `eval/{env}/pass@{k}` | Pass-at-k (binary reward tasks only) |
| `eval/{env}/completion_len/mean` | Mean completion length |
| `eval/{env}/is_truncated/mean` | Fraction truncated |
| `eval/{env}/no_response/mean` | Fraction of empty responses |
| `eval/{env}/failed_rollouts` | Rollout failure rate |
| `eval/{env}/time` | Eval wall time |

---

## System (`system/`)

| Metric | Description |
|---|---|
| `system/ckpt_disk_free_gib` | Free disk space for checkpoints (GiB) |
| `system/ckpt_disk_used_gib` | Used disk space for checkpoints (GiB) |
| `system/ckpt_disk_total_gib` | Total disk space (GiB) |
| `system/ckpt_disk_free_ratio` | Free disk ratio (0–1) |

---

## Tables (W&B Artifacts)

| Key | Description |
|---|---|
| `samples` | Cumulative rollout table (appended every `log_extras.interval` steps, default 10): prompt, completion, reward. Uses W&B incremental mode — rows accumulate across the run. |
| `eval/samples` | Cumulative eval rollout table, appended after each eval. |
| `final-samples` | Snapshot of all samples logged at end of run. |
