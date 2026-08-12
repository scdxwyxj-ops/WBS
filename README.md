# WBS SAM2 Inference

Lightweight SAM2 inference for bbox-guided white-blood-cell segmentation.

## Installation

Use Python 3.11. For CUDA inference, install the PyTorch 2.8 build matching your CUDA environment first.

```bash
pip install -e ".[sam2]"
python scripts/download_checkpoint.py --output checkpoints/sam2.1_hiera_large.pt
```

## Inference

```bash
wbs-infer \
  --config configs/default.json \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --image path/to/image.png \
  --bbox 120 80 460 430 \
  --output outputs/mask.png
```

The output is a binary mask at the original image size. Run metadata is saved next to the mask.

The proposal-retention parameter `n` is `selection.retain_n` in [configs/default.json](configs/default.json). Its default value is `2`.

## Python API

```python
import numpy as np
from PIL import Image
from wbs_inference import WBSSegmenter

segmenter = WBSSegmenter.from_checkpoint(
    "checkpoints/sam2.1_hiera_large.pt",
    "configs/default.json",
)
image = np.asarray(Image.open("cell.png").convert("RGB"))
result = segmenter.predict(image, (120, 80, 460, 430))
result.save("outputs/mask.png", "outputs/mask.json")
```

## Tests

```bash
pip install -e ".[test]"
pytest
ruff check .
```

## License

Apache-2.0. SAM2 is installed from the pinned upstream revision in `pyproject.toml`.
