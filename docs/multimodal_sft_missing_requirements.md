# Multimodal SFT — Missing Requirements

The RL trainer already has full multimodal support. This document tracks what needs to be added to the SFT path to reach parity.

## Current state

| Component | RL | SFT |
|-----------|-----|-----|
| Model forward (pixel_values, image_grid_thw, mm_token_type_ids) | ✅ `trainer/model.py:874-913` | ✅ (shared) |
| FSDP vision encoder sharding | ✅ `trainer/model.py:401-453` | ✅ (shared) |
| VLM config (`model.vlm`) | ✅ | ✅ |
| Data structures include multimodal fields | ✅ `trainer/rl/data.py` | ❌ |
| Dataset extracts and processes images | ✅ `orchestrator/trajectories.py` | ❌ |
| Collation handles variable-size pixel tensors | ✅ | ❌ |
| `compute_loss()` passes multimodal tensors | ✅ `trainer/rl/train.py` | ❌ |
| Image processor loaded alongside tokenizer | ✅ | ❌ |
| Position IDs omitted for VLM (MRoPE) | ✅ | ❌ |
| Packing disabled for multimodal samples | ✅ | ❌ |

## Required changes

### 1. `src/prime_rl/trainer/sft/data.py` — most of the work

**`Sample` and `Batch` TypedDicts** need multimodal fields added, mirroring `TensorMicroBatch` in `trainer/rl/data.py:16-40`:

```python
class Sample(TypedDict, total=False):
    # existing fields ...
    pixel_values: bytes          # serialised like RL transport layer
    image_grid_thw: list[list[int]]
    mm_token_type_ids: list[int]
```

**`SFTDataset._process()`** needs to:
- Extract images from `example['messages']` (base64 or PIL)
- Run them through an image processor to produce `pixel_values` and `image_grid_thw`
- Compute `mm_token_type_ids` by tracking which token positions came from images
- Skip returning `position_ids` when images are present — VLM models compute MRoPE internally (the shared `model.forward()` at `model.py:894-902` already branches on this)

**`SFTDataset.__init__()`** needs a `processor` parameter (image processor, see item 3 below).

**`stack_collate()` / `cat_collate()`** need to handle variable-size `pixel_values`. The simplest approach (matching RL) is to disable packing entirely for multimodal batches and use a plain collate that converts fields to tensors. See `trainer/rl/data.py:205-215` for the RL reference.

### 2. `src/prime_rl/trainer/sft/train.py`

**`compute_loss()`** (lines 207–239) only passes `input_ids` and `position_ids` to the model. It needs to:

1. Extract `pixel_values`, `image_grid_thw`, `mm_token_type_ids` from the micro-batch
2. Move them to CUDA
3. Pass them as kwargs to `forward()`
4. Handle the position-ID omission for VLM (pass `None` or omit)

Context parallelism (CP) sharding of pixel tensors: images should likely skip CP sharding (pixel values aren't sequence-parallel). Follow the pattern in `trainer/rl/train.py:354-408` but exclude multimodal tensors from the CP scatter.

### 3. `src/prime_rl/trainer/model.py` — `setup_tokenizer()`

Currently only loads a tokenizer. When `config.vlm` is set, it should load `AutoProcessor` instead so the image processor is available:

```python
if config.vlm:
    processor = AutoProcessor.from_pretrained(config.name)
    return processor.tokenizer, processor.image_processor
else:
    return AutoTokenizer.from_pretrained(config.name), None
```

The image processor then needs to be threaded through to `setup_dataset()` in the SFT trainer.

### 4. Tests and example config

- **Integration test**: add a multimodal case to `tests/integration/test_sft.py`
- **Unit tests**: `tests/unit/train/sft/` — at minimum test multimodal collation and the position-ID omission logic
- **CI config**: create `configs/ci/integration/sft/multimodal_start.toml` using a Qwen3-VL model

## Implementation order

1. Extend `Sample` / `Batch` TypedDicts
2. Update `SFTDataset._process()` to extract images and compute multimodal fields
3. Wire image processor into `setup_dataset()` / `SFTDataset.__init__()`
4. Update `compute_loss()` to pass multimodal tensors to `forward()`
5. Fix position-ID handling (omit for VLM)
6. Disable packing for multimodal batches
7. Update `setup_tokenizer()` to load `AutoProcessor` for VLM
8. Add tests and example config

## Key reference points

- VLM utilities: `src/prime_rl/utils/vlm.py`
- RL multimodal data structures: `src/prime_rl/trainer/rl/data.py`
- Shared model forward with multimodal support: `src/prime_rl/trainer/model.py:874-913`
- Working RL multimodal config: `configs/multimodal/rl_color_codeword.toml`
- Image caching in RL orchestrator: `src/prime_rl/orchestrator/trajectories.py` (`VLMImageCache`)
