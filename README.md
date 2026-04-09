# Monocular Depth Estimation Benchmarks

Benchmarks for monocular depth estimation models on the Replica dataset (office0 scene).

## Usage

```bash
# Latency-only benchmark
python eval_latency.py --mode hf_baselines \
  --input-dir datasets/Replica/office0/results \
  --prefix frame --num-images 100 --warmup 5 --no-save

# Evaluation with accuracy metrics (against GT depth)
python eval_depth.py --mode hf_baselines \
  --input-dir datasets/Replica/office0/results \
  --prefix frame --num-images 100 --warmup 5
```

## Results

Replica office0, 100 images (1200x680), 5 warmup, GPU: NVIDIA A100 80GB PCIe

Relative-depth models are aligned to GT via least-squares scale+shift.

Pixel-level: AbsRel, SqRel, RMSE, RMSE_log (lower = better), δ1/δ2/δ3 (higher = better).
Structural: SSIM (higher = better), GradErr (lower = better), EdgeAcc/EdgeComp (higher = better), OrdErr (lower = better).

### HuggingFace Baselines

| Model | AbsRel | RMSE | δ<1.25 | SSIM | GradErr | EdgeAcc | EdgeComp | OrdErr | Mean (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DepthPro | 0.0192 | 0.0507 | 0.9996 | 0.9926 | 0.2253 | 0.7998 | 0.9363 | 0.0266 | 318.0 | 231.6 | 451.6 |
| ZoeDepth | 0.0725 | 0.1929 | 0.9867 | 0.9738 | 0.2819 | 0.6712 | 0.7103 | 0.1035 | 1715.1 | 984.2 | 2378.8 |
| DA-V2-Small | 0.0911 | 0.2276 | 0.9506 | 0.9791 | 0.2576 | 0.5059 | 0.5236 | 0.0720 | 110.5 | 46.6 | 167.9 |
| DPT-Large | 0.0984 | 0.2837 | 0.9150 | 0.9728 | 0.2657 | 0.5033 | 0.4521 | 0.1568 | 157.2 | 55.3 | 266.7 |
| DPT-Hybrid | 0.0990 | 0.2866 | 0.9145 | 0.9730 | 0.2627 | 0.5401 | 0.5063 | 0.1548 | 130.1 | 35.4 | 262.6 |

### Depth Anything 3

| Model | AbsRel | RMSE | δ<1.25 | SSIM | GradErr | EdgeAcc | EdgeComp | OrdErr | Mean (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DA3NESTED-GIANT-LARGE | 0.0053 | 0.0343 | 0.9983 | 0.9879 | 0.2623 | 0.9665 | 0.9024 | 0.0081 | 137.4 | 126.6 | 173.3 |
| DA3-GIANT | 0.0060 | 0.0347 | 0.9983 | 0.9880 | 0.2610 | 0.9652 | 0.9007 | 0.0086 | 115.8 | 101.8 | 142.6 |
| DA3-LARGE | 0.0100 | 0.0419 | 0.9976 | 0.9869 | 0.2639 | 0.9365 | 0.8812 | 0.0147 | 84.9 | 63.9 | 107.3 |
| DA3-BASE | 0.0140 | 0.0493 | 0.9978 | 0.9859 | 0.2640 | 0.8567 | 0.7944 | 0.0206 | 65.7 | 48.6 | 86.0 |
| DA3-SMALL | 0.0255 | 0.0781 | 0.9956 | 0.9839 | 0.2645 | 0.7675 | 0.7103 | 0.0378 | 64.8 | 52.9 | 85.1 |
| DA3METRIC-LARGE | 0.0340 | 0.0881 | 0.9987 | 0.9873 | 0.2561 | 0.9112 | 0.8244 | 0.0452 | 62.3 | 39.4 | 79.8 |

### NVIDIA TAO Toolkit (Replica office0, 100 images)

| Model | Mean (ms) |
|---|---:|
| NvDepthAnythingV2-Large | 649.5 |

TAO latency includes Docker container startup overhead averaged over 100 images.
