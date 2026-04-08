# Monocular Depth Estimation Benchmarks

Benchmarks for monocular depth estimation models on sample images (800x600).

## Usage

```bash
python run_depth_anything.py --mode hf_baselines
python run_depth_anything.py --mode da3
python run_depth_anything.py --mode tao
```

## Results

GPU: NVIDIA A100 (80GB)

### HuggingFace Baselines

| Model | Mean (s) | Min (s) | Max (s) |
|---|---:|---:|---:|
| Depth-Anything-V2-Small | 0.1412 | 0.0450 | 0.3762 |
| DPT-Large | 0.1457 | 0.0871 | 0.2092 |
| DPT-Hybrid-MiDaS | 0.2007 | 0.0752 | 0.3205 |
| DepthPro | 0.3247 | 0.2266 | 0.5330 |
| ZoeDepth | 1.4695 | 1.0051 | 1.9055 |

### Depth Anything 3 (New)

| Model | Mean (s) | Min (s) | Max (s) |
|---|---:|---:|---:|
| DA3-SMALL | 0.0754 | 0.0582 | 0.1011 |
| DA3-BASE | 0.0712 | 0.0626 | 0.0888 |
| DA3-LARGE-1.1 | 0.0854 | 0.0829 | 0.0928 |
| DA3-GIANT-1.1 | 0.1187 | 0.1108 | 0.1260 |
| DA3NESTED-GIANT-LARGE-1.1 | 0.2611 | 0.1528 | 0.7541 |

### NVIDIA TAO Toolkit

| Model | Mean (s) |
|---|---:|
| NvDepthAnythingV2-Large | 8.4103 |

Note: TAO latency includes Docker container startup.