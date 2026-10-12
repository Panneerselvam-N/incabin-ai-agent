# VLM Development Base

Modular prefix-fusion Vision Language Model (VLM) scaffold. It follows the reference architecture: a vision encoder produces patch features, a custom projector maps them to the language-model hidden space, and the projected image tokens are prepended to text embeddings.

## Layout

```text
configs/                  Hyperparameters and model selection
data/                     JSONL annotations and image directory
src/vlm_dev/
  models/                 Vision adapter, LM adapter, projector, VLM fusion model
  data/                   Dataset and collator
  training/               Trainer and validator
  utils/                  Config, seed, checkpoint helpers
  scripts/                CLI training and evaluation entry points
```

## Install

```bash
uv sync
```

## Dataset format

Place images in `data/images/` and create one JSON object per line in `data/train.jsonl` and `data/validation.jsonl`:

```json
{"image": "frame_0001.jpg", "prompt": "Describe the driving scene.", "answer": "A car is driving on a city road."}
```

The target answer alone contributes to loss; prompt, padding, and visual-prefix tokens are masked. Update dataset paths and hyperparameters in `configs/siglip2_qwen2_5.yaml`.

## Precision configuration

Set `model.model_dtype` to `bfloat16` (recommended on supported CUDA devices), `float16`, `float32`, or `auto` (retain the checkpoint dtype). The vision encoder input, projector input, projector output, and language-model embeddings are cast at each module boundary. Keep `training.mixed_precision` compatible with the selected model dtype; use `no` with `float32`.

## Train

```bash
uv run python -m vlm_dev.scripts.train --config configs/siglip2_qwen2_5.yaml
```

For multi-GPU training:

```bash
uv run accelerate launch -m vlm_dev.scripts.train --config configs/siglip2_qwen2_5.yaml
```

## Evaluate

```bash
uv run python -m vlm_dev.scripts.evaluate \
  --config configs/siglip2_qwen2_5.yaml \
  --checkpoint outputs/siglip2-qwen2.5/last
```

## Inference

```bash
uv run python -m vlm_dev.scripts.infer \
  --config configs/siglip2_qwen2_5.yaml \
  --checkpoint outputs/siglip2-qwen2.5/last \
  --image data/images/frame_0001.jpg \
  --prompt "Describe the driving scene."
```

Add `--do-sample --temperature 0.7 --top-p 0.9` for sampled output; greedy decoding is the default.

For a pre-training smoke test, omit `--checkpoint`. This loads the pretrained vision and language backbones with a random custom projector, so output is expected to be visually ungrounded.

```bash
uv run python -m vlm_dev.scripts.infer \
  --config configs/siglip2_qwen2_5.yaml \
  --image data/images/frame_0001.jpg
```

Use it in Python:

```python
from vlm_dev.utils.config import load_config
from vlm_dev.utils.inference import VLMInference

engine = VLMInference(load_config("configs/siglip2_qwen2_5.yaml"), "outputs/siglip2-qwen2.5/last")
answer = engine.generate("data/images/frame_0001.jpg", "Describe the driving scene.")
```

## Extension points

- Vision encoder: `src/vlm_dev/models/vision_encoder.py`
- Qwen or other causal LM: `src/vlm_dev/models/language_model.py`
- Custom projector variants: `src/vlm_dev/models/projector.py`
- Multimodal token fusion: `src/vlm_dev/models/vlm.py`
- Dataset schemas/transforms: `src/vlm_dev/data/dataset.py`
