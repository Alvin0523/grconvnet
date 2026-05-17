---
icon: lucide/brain
---

# Model — GR-ConvNet v2

## What it is

GR-ConvNet v2 (`grconvnet3` in code) is a fully convolutional encoder-decoder network for robotic grasp detection. It takes a depth image as input and outputs three pixel-aligned maps in a single forward pass (~20ms on GPU).

**Repo**: `robotic-grasping/` (cloned from `skumra/robotic-grasping`)

---

## Architecture

```
Input (1×224×224 depth image)
      ↓
Conv1 9×9 → BN → ReLU           [32 ch]
Conv2 4×4 stride 2 → BN → ReLU  [64 ch]
Conv3 4×4 stride 2 → BN → ReLU  [128 ch]
      ↓
ResBlock × 5                     [128 ch each]
      ↓
ConvTranspose4 4×4 stride 2      [64 ch]
ConvTranspose5 4×4 stride 2      [32 ch]
ConvTranspose6 9×9               [32 ch]
      ↓
4× output heads (2×2 conv):
  pos_output   → Quality map Q
  cos_output   → cos(2Φ) angle map
  sin_output   → sin(2Φ) angle map
  width_output → Width map W
```

Total params: ~1.9M · Model size: ~7.2MB

The angle is decomposed into cos/sin to eliminate the ±π/2 wrap-around discontinuity that would otherwise confuse the network.

---

## Code structure (`robotic-grasping/`)

```
robotic-grasping/
├── inference/
│   ├── models/
│   │   ├── grasp_model.py     base class + ResidualBlock definition
│   │   ├── grconvnet3.py      GR-ConvNet v2 — the one we use
│   │   ├── grconvnet.py       original GR-ConvNet (unused)
│   │   ├── grconvnet2.py      multi-dropout variant (unused)
│   │   └── grconvnet4.py      inverted variant (unused)
│   ├── grasp_generator.py     runs inference, returns grasp candidates
│   └── post_process.py        peak detection on Q map, decode angle
├── utils/
│   ├── data/
│   │   ├── jacquard_data.py   Jacquard dataset loader
│   │   ├── cornell_data.py    Cornell dataset loader
│   │   └── grasp_data.py      base dataset class
│   └── dataset_processing/
│       ├── grasp.py           GraspRectangle class, IOU computation
│       ├── evaluation.py      IOU evaluation metrics
│       └── image.py           DepthImage, Image with augmentation
├── hardware/
│   ├── camera.py              RealSense camera interface
│   └── device.py              GPU/CPU device selection
├── trained-models/            pre-trained checkpoints (see below)
├── train_network.py           training entry point
└── evaluate.py                evaluation entry point
```

---

## Pre-trained models

| Model | Dataset | IOU | Input | Notes |
|---|---|---|---|---|
| `cornell-randsplit-rgbd-grconvnet3-drop1-ch32` | Cornell | 98% | RGBD 224px | dropout=1 |
| `cornell-randsplit-rgbd-grconvnet3-drop1-ch16` | Cornell | 97% | RGBD 224px | lighter |
| `jacquard-d-grconvnet3-drop0-ch32` | Jacquard V1 (depth) | 94% | D 300px | **use as baseline** |
| `jacquard-rgbd-grconvnet3-drop0-ch32` | Jacquard V1 (RGBD) | 93% | RGBD 300px | |

We train a new model on **Jacquard V2 (depth-only)** — expected to exceed 94% IOU.

---

## Training on Jacquard V2

```bash
pixi run train
```

Which runs:

```bash
python robotic-grasping/train_network.py \
  --dataset jacquard \
  --dataset-path data/jacquard \
  --use-depth 1 --use-rgb 0 \
  --use-dropout 0 --input-size 300 \
  --epochs 50 --batch-size 8 \
  --logdir output/models \
  --description grconvnet_jacquard
```

Key flags for Jacquard:

| Flag | Value | Why |
|---|---|---|
| `--use-depth 1 --use-rgb 0` | depth only | matches pre-trained baseline; simpler = faster |
| `--use-dropout 0` | no dropout | Jacquard is large enough that dropout hurts |
| `--input-size 300` | 300px | Jacquard images are 1024px — 300 is the recommended crop |
| `--network grconvnet3` | default | GR-ConvNet v2 architecture |

Training is tracked with W&B (`wandb`). Run `wandb login` once before training.

---

## Jacquard V2 toolkit (`Jacquard_V2/`)

```
Jacquard_V2/
└── Jacquard_V2/
    ├── train.py           alternative trainer built for V2
    ├── eval.py            evaluation script
    ├── add_fn_data.py     HIL false-negative annotation tool
    ├── utils/             V2-specific data loader
    └── models/            benchmarked architectures (GG-CNN2, MobileNetV2, etc.)
```

The `robotic-grasping/` pipeline is what connects to DISCOVERSE, so we train there. The `Jacquard_V2/` toolkit is useful for reference and for running the V2-specific eval.
