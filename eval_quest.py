import argparse
import csv
import os
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, distance_transform_edt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def slugify(name: str) -> str:
    return name.split("/")[-1].removesuffix("-hf")

INVERT_MODELS = {
    "depth-anything/Depth-Anything-V2-Small-hf",
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
}

METRIC_MODELS = {
    "apple/DepthPro-hf",
    "depth-anything/DA3METRIC-LARGE",
}

DA3_MODELS = {
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "depth-anything/DA3-GIANT-1.1",
    "depth-anything/DA3-LARGE-1.1",
    "depth-anything/DA3-BASE",
    "depth-anything/DA3-SMALL",
    "depth-anything/DA3METRIC-LARGE",
}

RESULT_FIELDS = [
    "model", "num_images",
    "abs_rel", "sq_rel", "rmse", "rmse_log",
    "delta1", "delta2", "delta3",
    "ssim", "grad_err", "edge_acc", "edge_comp", "ord_err",
    "mean_latency_ms", "min_latency_ms", "max_latency_ms",
]


def load_pairs(dataset_dir: str, num_images: int) -> list[dict]:
    meta_path = os.path.join(dataset_dir, "metadata.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")
    pairs = []
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            rgb_path = os.path.join(dataset_dir, "rgb",      f"{row['rgb_ts']}.png")
            gt_path  = os.path.join(dataset_dir, "depth", f"{row['depth_ts']}.npy")
            if os.path.isfile(rgb_path) and os.path.isfile(gt_path):
                pairs.append({"rgb": rgb_path, "gt": gt_path})
    if num_images > 0:
        pairs = pairs[:num_images]
    return pairs


def align_scale_shift(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p = pred[mask].flatten()
    g = gt[mask].flatten()
    s, t = np.linalg.lstsq(np.stack([p, np.ones_like(p)], axis=1), g, rcond=None)[0]
    return s * pred + t


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    p = np.clip(pred[mask], 1e-6, None)
    g = gt[mask]
    thresh = np.maximum(p / g, g / p)
    return dict(
        abs_rel  = float(np.mean(np.abs(p - g) / g)),
        sq_rel   = float(np.mean((p - g) ** 2 / g)),
        rmse     = float(np.sqrt(np.mean((p - g) ** 2))),
        rmse_log = float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        delta1   = float((thresh < 1.25).mean()),
        delta2   = float((thresh < 1.25 ** 2).mean()),
        delta3   = float((thresh < 1.25 ** 3).mean()),
        ssim     = _ssim(pred, gt, mask),
        grad_err = _grad_err(pred, gt, mask),
        **dict(zip(("edge_acc", "edge_comp"), _edge_metrics(pred, gt, mask))),
        ord_err  = _ordinal_err(pred, gt, mask),
    )


def _ssim(pred, gt, mask, sigma=1.5):
    L   = max(gt[mask].max() - gt[mask].min(), 1e-6)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_p = gaussian_filter(pred, sigma);  mu_g = gaussian_filter(gt, sigma)
    s2_p = gaussian_filter(pred ** 2, sigma) - mu_p ** 2
    s2_g = gaussian_filter(gt   ** 2, sigma) - mu_g ** 2
    s_pg = gaussian_filter(pred * gt, sigma) - mu_p * mu_g
    num  = (2 * mu_p * mu_g + C1) * (2 * s_pg + C2)
    den  = (mu_p ** 2 + mu_g ** 2 + C1) * (s2_p + s2_g + C2)
    return float((num / den)[mask].mean())


def _grad_err(pred, gt, mask):
    err = (sobel(pred, 1)[mask] - sobel(gt, 1)[mask]) ** 2 + \
          (sobel(pred, 0)[mask] - sobel(gt, 0)[mask]) ** 2
    return float(np.sqrt(np.mean(err)))


def _edge_metrics(pred, gt, mask, pct=90, tol=3):
    gt_mag   = np.sqrt(sobel(gt,   0) ** 2 + sobel(gt,   1) ** 2)
    pred_mag = np.sqrt(sobel(pred, 0) ** 2 + sobel(pred, 1) ** 2)
    gt_e   = (gt_mag   > np.percentile(gt_mag[mask],   pct)) & mask
    pred_e = (pred_mag > np.percentile(pred_mag[mask], pct)) & mask
    if not gt_e.any() or not pred_e.any():
        return 0.0, 0.0
    acc  = float((distance_transform_edt(~gt_e)[pred_e]   <= tol).mean())
    comp = float((distance_transform_edt(~pred_e)[gt_e]   <= tol).mean())
    return acc, comp


def _ordinal_err(pred, gt, mask, n=50000):
    ys, xs = np.where(mask)
    if len(ys) < 2:
        return 0.0
    rng = np.random.RandomState(42)
    idx = rng.choice(len(ys), size=(n, 2), replace=True)
    gt_d   = gt[ys[idx[:, 0]],   xs[idx[:, 0]]]   - gt[ys[idx[:, 1]],   xs[idx[:, 1]]]
    pred_d = pred[ys[idx[:, 0]], xs[idx[:, 0]]] - pred[ys[idx[:, 1]], xs[idx[:, 1]]]
    valid  = np.abs(gt_d) > 1e-6
    return float((gt_d[valid] * pred_d[valid] < 0).mean()) if valid.any() else 0.0


def predict_hf(pipe, rgb_path: str, invert: bool) -> tuple[np.ndarray, float]:
    img  = Image.open(rgb_path).convert("RGB")
    t0   = time.perf_counter()
    pred = np.array(pipe(img)["depth"]).astype(np.float64)
    lat  = (time.perf_counter() - t0) * 1000
    if invert:
        pred = pred.max() - pred
    return pred, lat


def predict_da3(model, rgb_path: str) -> tuple[np.ndarray, float]:
    t0  = time.perf_counter()
    out = model.inference([rgb_path])
    lat = (time.perf_counter() - t0) * 1000
    return out.depth[0].astype(np.float64), lat


def resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.array(
        Image.fromarray(arr.astype(np.float32), mode="F").resize((w, h), resample=Image.BILINEAR)
    )


def evaluate(pairs: list[dict], get_pred, save_dir: str | None,
             metric_save_dir: str | None = None) -> dict:
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if metric_save_dir:
        os.makedirs(metric_save_dir, exist_ok=True)
    all_metrics, latencies = [], []
    for p in pairs:
        pred, lat = get_pred(p["rgb"])
        latencies.append(lat)
        ts   = os.path.splitext(os.path.basename(p["rgb"]))[0]
        gt   = np.load(p["gt"]).astype(np.float64)
        mask = gt > 1e-3
        if metric_save_dir:
            np.save(os.path.join(metric_save_dir, f"{ts}.npy"), pred.astype(np.float32))
        if pred.shape != gt.shape:
            pred = resize_to(pred, gt.shape[0], gt.shape[1])
        pred_aligned = align_scale_shift(pred, gt, mask)
        all_metrics.append(compute_metrics(pred_aligned, gt, mask))
        if save_dir:
            np.save(os.path.join(save_dir, f"{ts}.npy"), pred_aligned.astype(np.float32))
    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    avg["mean_latency_ms"] = float(np.mean(latencies))
    avg["min_latency_ms"]  = float(np.min(latencies))
    avg["max_latency_ms"]  = float(np.max(latencies))
    avg["num_images"]      = len(pairs)
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model")
    parser.add_argument("--load-preds", action="store_true")
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required (use with --save-preds to run inference, or --load-preds to reload saved predictions)")

    pairs = load_pairs(args.dataset_dir, args.num_images)
    if not pairs:
        raise ValueError(f"No pairs found in {args.dataset_dir}")
    print(f"Loaded {len(pairs)} pairs")

    model_label       = args.model
    model_slug        = slugify(model_label)
    pred_depth_dir    = os.path.join(args.dataset_dir, "pred_depth", model_slug)
    metric_depth_dir  = (
        os.path.join(args.dataset_dir, "pred_depth_metric", model_slug)
        if args.save_preds and model_label in METRIC_MODELS
        else None
    )

    if args.load_preds:
        missing = [p for p in pairs if not os.path.isfile(
            os.path.join(pred_depth_dir, os.path.splitext(os.path.basename(p["rgb"]))[0] + ".npy"))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} pred_depth files missing in {pred_depth_dir}")
        def get_pred(rgb_path):
            ts = os.path.splitext(os.path.basename(rgb_path))[0]
            return np.load(os.path.join(pred_depth_dir, f"{ts}.npy")).astype(np.float64), 0.0
    elif args.model in DA3_MODELS:
        from depth_anything_3.api import DepthAnything3
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = DepthAnything3.from_pretrained(args.model).to(device=device)
        for i in range(args.warmup):
            model.inference([pairs[i % len(pairs)]["rgb"]])
        print(f"[{args.model}] warmup done")
        def get_pred(rgb_path):
            return predict_da3(model, rgb_path)
    else:
        import torch
        from transformers import pipeline as hf_pipeline
        invert = args.model in INVERT_MODELS
        device = 0 if torch.cuda.is_available() else -1
        pipe   = hf_pipeline(task="depth-estimation", model=args.model, device=device)
        for i in range(args.warmup):
            predict_hf(pipe, pairs[i % len(pairs)]["rgb"], invert)
        print(f"[{args.model}] warmup done")
        def get_pred(rgb_path):
            return predict_hf(pipe, rgb_path, invert)

    results = evaluate(pairs, get_pred, pred_depth_dir if args.save_preds else None,
                       metric_depth_dir)
    results["model"] = model_label

    session  = os.path.basename(os.path.normpath(args.dataset_dir))
    out_dir  = os.path.join(PROJECT_ROOT, "outputs", "quest", session)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "eval_results.csv")

    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: f"{results[k]:.4f}" if isinstance(results[k], float) else results[k]
                    for k in RESULT_FIELDS})

    print(f"\nResults → {csv_path}")
    print(f"\n{'Model':<45} {'AbsRel':>8} {'RMSE':>8} {'δ<1.25':>8} {'SSIM':>8} {'Lat(ms)':>9}")
    print("-" * 95)
    print(f"{model_label:<45} {results['abs_rel']:8.4f} {results['rmse']:8.4f} "
          f"{results['delta1']:8.4f} {results['ssim']:8.4f} {results['mean_latency_ms']:9.1f}")


if __name__ == "__main__":
    main()
