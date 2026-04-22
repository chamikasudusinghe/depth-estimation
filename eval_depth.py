"""Evaluate predicted depth maps against Replica ground truth.

Usage:
  python eval_depth.py --mode hf_baselines \
    --input-dir datasets/Replica/office0/results \
    --prefix frame --num-images 100 --warmup 5
"""

import argparse
import csv
import os
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, distance_transform_edt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEPTH_SCALE = 6553.5

HF_BASELINES = [
    "depth-anything/Depth-Anything-V2-Small-hf",
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
    "Intel/zoedepth-nyu-kitti",
    "apple/DepthPro-hf",
]

DA3_MODELS = [
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "depth-anything/DA3-GIANT-1.1",
    "depth-anything/DA3-LARGE-1.1",
    "depth-anything/DA3-BASE",
    "depth-anything/DA3-SMALL",
    "depth-anything/DA3METRIC-LARGE",
]

INVERT_MODELS = {
    "depth-anything/Depth-Anything-V2-Small-hf",
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
}


def slugify(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def collect_image_pairs(input_dir: str, prefix: str, num_images: int):
    frames = sorted(
        f for f in os.listdir(input_dir)
        if f.startswith(prefix) and f.endswith(".jpg")
    )
    if num_images > 0:
        frames = frames[:num_images]
    pairs = []
    for f in frames:
        idx = f.replace(prefix, "").replace(".jpg", "")
        depth_name = f"depth{idx}.png"
        depth_path = os.path.join(input_dir, depth_name)
        if os.path.isfile(depth_path):
            pairs.append((os.path.join(input_dir, f), depth_path))
    return pairs


def load_gt_depth(path: str) -> np.ndarray:
    arr = np.array(Image.open(path)).astype(np.float64)
    return arr / DEPTH_SCALE


def align_scale_shift(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """Least-squares: find s, t such that s*pred + t ≈ gt (on valid pixels)."""
    p = pred[mask].flatten()
    g = gt[mask].flatten()
    A = np.stack([p, np.ones_like(p)], axis=1)
    result = np.linalg.lstsq(A, g, rcond=None)
    s, t = result[0]
    return s * pred + t


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    p = np.clip(pred[mask], 1e-6, None)
    g = gt[mask]

    thresh = np.maximum(p / g, g / p)
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(p - g) / g)
    sq_rel = np.mean((p - g) ** 2 / g)
    rmse = np.sqrt(np.mean((p - g) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))

    ssim = compute_ssim(pred, gt, mask)
    grad_err = compute_gradient_error(pred, gt, mask)
    edge_acc, edge_comp = compute_edge_metrics(pred, gt, mask)
    ord_err = compute_ordinal_error(pred, gt, mask)

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "delta1": d1,
        "delta2": d2,
        "delta3": d3,
        "ssim": ssim,
        "grad_err": grad_err,
        "edge_acc": edge_acc,
        "edge_comp": edge_comp,
        "ord_err": ord_err,
    }


def compute_ssim(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                 sigma: float = 1.5) -> float:
    L = max(gt[mask].max() - gt[mask].min(), 1e-6)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    mu_p = gaussian_filter(pred, sigma)
    mu_g = gaussian_filter(gt, sigma)
    sigma_p2 = gaussian_filter(pred ** 2, sigma) - mu_p ** 2
    sigma_g2 = gaussian_filter(gt ** 2, sigma) - mu_g ** 2
    sigma_pg = gaussian_filter(pred * gt, sigma) - mu_p * mu_g

    num = (2 * mu_p * mu_g + C1) * (2 * sigma_pg + C2)
    den = (mu_p ** 2 + mu_g ** 2 + C1) * (sigma_p2 + sigma_g2 + C2)
    ssim_map = num / den
    return float(ssim_map[mask].mean())


def compute_gradient_error(pred: np.ndarray, gt: np.ndarray,
                           mask: np.ndarray) -> float:
    pred_dx, pred_dy = sobel(pred, axis=1), sobel(pred, axis=0)
    gt_dx, gt_dy = sobel(gt, axis=1), sobel(gt, axis=0)
    err = (pred_dx[mask] - gt_dx[mask]) ** 2 + (pred_dy[mask] - gt_dy[mask]) ** 2
    return float(np.sqrt(np.mean(err)))


def compute_edge_metrics(pred: np.ndarray, gt: np.ndarray,
                         mask: np.ndarray, pct: float = 90,
                         tolerance: int = 3) -> tuple[float, float]:
    gt_mag = np.sqrt(sobel(gt, 0) ** 2 + sobel(gt, 1) ** 2)
    pred_mag = np.sqrt(sobel(pred, 0) ** 2 + sobel(pred, 1) ** 2)

    gt_edges = (gt_mag > np.percentile(gt_mag[mask], pct)) & mask
    pred_edges = (pred_mag > np.percentile(pred_mag[mask], pct)) & mask

    if gt_edges.sum() == 0 or pred_edges.sum() == 0:
        return 0.0, 0.0

    gt_dist = distance_transform_edt(~gt_edges)
    pred_dist = distance_transform_edt(~pred_edges)

    accuracy = float((gt_dist[pred_edges] <= tolerance).mean())
    completeness = float((pred_dist[gt_edges] <= tolerance).mean())
    return accuracy, completeness


def compute_ordinal_error(pred: np.ndarray, gt: np.ndarray,
                          mask: np.ndarray, n_pairs: int = 50000) -> float:
    ys, xs = np.where(mask)
    if len(ys) < 2:
        return 0.0
    rng = np.random.RandomState(42)
    idx = rng.choice(len(ys), size=(n_pairs, 2), replace=True)
    gt_diff = gt[ys[idx[:, 0]], xs[idx[:, 0]]] - gt[ys[idx[:, 1]], xs[idx[:, 1]]]
    pred_diff = pred[ys[idx[:, 0]], xs[idx[:, 0]]] - pred[ys[idx[:, 1]], xs[idx[:, 1]]]
    valid = np.abs(gt_diff) > 1e-6
    if valid.sum() == 0:
        return 0.0
    return float((gt_diff[valid] * pred_diff[valid] < 0).mean())


def run_hf_eval(pairs, model_name: str, warmup: int):
    from transformers import pipeline as hf_pipeline

    invert = model_name in INVERT_MODELS
    pipe = hf_pipeline(task="depth-estimation", model=model_name)

    for i in range(warmup):
        img = Image.open(pairs[i % len(pairs)][0]).convert("RGB")
        pipe(img)
    print(f"[{model_name}] warmup {warmup} done")

    all_metrics = []
    latencies = []
    for rgb_path, gt_path in pairs:
        img = Image.open(rgb_path).convert("RGB")
        t0 = time.perf_counter()
        pred_pil = pipe(img)["depth"]
        lat = time.perf_counter() - t0
        latencies.append(lat)

        pred = np.array(pred_pil).astype(np.float64)
        if invert:
            pred = pred.max() - pred
        gt = load_gt_depth(gt_path)
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred.astype(np.float32), mode="F").resize(
                    (gt.shape[1], gt.shape[0]), resample=Image.BILINEAR
                )
            )

        mask = gt > 1e-3
        pred_aligned = align_scale_shift(pred, gt, mask)
        metrics = compute_metrics(pred_aligned, gt, mask)
        all_metrics.append(metrics)

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    avg["mean_latency"] = np.mean(latencies)
    avg["std_latency"] = np.std(latencies)
    avg["min_latency"] = np.min(latencies)
    avg["max_latency"] = np.max(latencies)
    avg["num_images"] = len(pairs)
    print(f"[{model_name}] eval done ({len(pairs)} images)")
    return avg


def run_da3_eval(pairs, model_name: str, warmup: int):
    from depth_anything_3.api import DepthAnything3  # type: ignore
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_name).to(device=device)

    for i in range(warmup):
        model.inference([pairs[i % len(pairs)][0]])
    print(f"[{model_name}] warmup {warmup} done")

    all_metrics = []
    latencies = []
    for rgb_path, gt_path in pairs:
        t0 = time.perf_counter()
        prediction = model.inference([rgb_path])
        lat = time.perf_counter() - t0
        latencies.append(lat)

        pred = prediction.depth[0].astype(np.float64)
        gt = load_gt_depth(gt_path)
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred.astype(np.float32), mode="F").resize(
                    (gt.shape[1], gt.shape[0]), resample=Image.BILINEAR
                )
            )

        mask = gt > 1e-3
        pred_aligned = align_scale_shift(pred, gt, mask)
        metrics = compute_metrics(pred_aligned, gt, mask)
        all_metrics.append(metrics)

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    avg["mean_latency"] = np.mean(latencies)
    avg["std_latency"] = np.std(latencies)
    avg["min_latency"] = np.min(latencies)
    avg["max_latency"] = np.max(latencies)
    avg["num_images"] = len(pairs)
    print(f"[{model_name}] eval done ({len(pairs)} images)")
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hf_baselines", "da3"], required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--prefix", default="frame")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    pairs = collect_image_pairs(args.input_dir, args.prefix, args.num_images)
    if not pairs:
        raise ValueError(f"No image/depth pairs found in {args.input_dir}")
    print(f"Found {len(pairs)} RGB-depth pairs")

    models = HF_BASELINES if args.mode == "hf_baselines" else DA3_MODELS
    run_fn = run_hf_eval if args.mode == "hf_baselines" else run_da3_eval

    results = []
    for m in models:
        try:
            r = run_fn(pairs, m, args.warmup)
            r["model"] = m
            results.append(r)
        except Exception as e:
            print(f"[{m}] SKIPPED: {e}")

    out_dir = os.path.join(PROJECT_ROOT, "outputs", args.mode)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "eval_results.csv")
    fields = ["model", "num_images", "abs_rel", "sq_rel", "rmse", "rmse_log",
              "delta1", "delta2", "delta3",
              "ssim", "grad_err", "edge_acc", "edge_comp", "ord_err",
              "mean_latency", "std_latency", "min_latency", "max_latency"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: f"{r[k]:.4f}" if isinstance(r[k], float) else r[k] for k in fields}
            w.writerow(row)

    print(f"\nResults saved to {csv_path}")
    print(f"\n{'Model':<45} {'AbsRel':>8} {'SSIM':>8} {'GradErr':>8} {'EdgeAcc':>8} {'EdgeCmp':>8} {'OrdErr':>8} {'Lat(s)':>8}")
    print("-" * 125)
    for r in sorted(results, key=lambda x: x["abs_rel"]):
        print(f"{r['model']:<45} {r['abs_rel']:8.4f} {r['ssim']:8.4f} {r['grad_err']:8.4f} "
              f"{r['edge_acc']:8.4f} {r['edge_comp']:8.4f} {r['ord_err']:8.4f} {r['mean_latency']:8.4f}")


if __name__ == "__main__":
    main()
