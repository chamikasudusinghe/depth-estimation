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

Metrics: AbsRel, SqRel, RMSE, RMSE_log (lower = better), δ1/δ2/δ3 (higher = better).
Relative-depth models are aligned to GT via least-squares scale+shift.

### HuggingFace Baselines

| Model | AbsRel | SqRel | RMSE | RMSE_log | δ<1.25 | δ<1.25² | δ<1.25³ | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DepthPro | 0.0192 | 0.0012 | 0.0507 | 0.0245 | 0.9996 | 1.0000 | 1.0000 | 337.3 | 61.7 | 232.3 | 465.7 |
| ZoeDepth | 0.0725 | 0.0174 | 0.1929 | 0.0903 | 0.9867 | 0.9990 | 0.9999 | 1798.1 | 315.9 | 982.0 | 2392.5 |
| DA-V2-Small | 0.0911 | 0.0254 | 0.2276 | 0.1268 | 0.9506 | 0.9912 | 0.9978 | 83.6 | 40.3 | 36.6 | 183.7 |
| DPT-Large | 0.0984 | 0.0360 | 0.2837 | 0.1245 | 0.9150 | 0.9990 | 0.9997 | 137.1 | 40.2 | 59.1 | 224.8 |
| DPT-Hybrid | 0.0990 | 0.0365 | 0.2866 | 0.1248 | 0.9145 | 0.9994 | 0.9998 | 144.5 | 44.7 | 50.6 | 240.7 |

### Depth Anything 3

| Model | AbsRel | SqRel | RMSE | RMSE_log | δ<1.25 | δ<1.25² | δ<1.25³ | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DA3NESTED-GIANT-LARGE-1.1 | 0.0053 | 0.0007 | 0.0343 | 0.0182 | 0.9983 | 0.9998 | 1.0000 | 131.2 | 8.0 | 116.4 | 152.8 |
| DA3-GIANT-1.1 | 0.0060 | 0.0007 | 0.0347 | 0.0187 | 0.9983 | 0.9998 | 1.0000 | 106.5 | 8.2 | 90.7 | 131.3 |
| DA3-LARGE-1.1 | 0.0100 | 0.0010 | 0.0419 | 0.0224 | 0.9976 | 0.9997 | 1.0000 | 76.0 | 7.1 | 63.8 | 94.2 |
| DA3-BASE | 0.0140 | 0.0013 | 0.0493 | 0.0251 | 0.9978 | 0.9998 | 1.0000 | 60.7 | 7.7 | 45.7 | 86.0 |
| DA3-SMALL | 0.0255 | 0.0035 | 0.0781 | 0.0400 | 0.9956 | 0.9991 | 1.0000 | 54.6 | 7.4 | 42.9 | 79.3 |

### NVIDIA TAO Toolkit (Replica office0, 100 images)

| Model | Mean (ms) |
|---|---:|
| NvDepthAnythingV2-Large | 649.5 |

TAO latency includes Docker container startup overhead averaged over 100 images.
