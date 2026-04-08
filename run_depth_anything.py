"""Monocular depth estimation benchmark.

Run modes:
  --mode hf_baselines
  --mode da3
  --mode tao
"""

import argparse
import csv
import os
import subprocess
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGES_DIR = "/home/chamika2/tao-projects/depth-estimation/images/sample5"
PROJECT_ROOT = "/home/chamika2/tao-projects/depth-estimation"

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
]

INVERT_MODELS = {
    "Intel/zoedepth-nyu-kitti",
    "apple/DepthPro-hf",
}


def slugify(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def collect_images(input_dir: str) -> list[str]:
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        os.path.join(input_dir, n)
        for n in os.listdir(input_dir)
        if os.path.splitext(n)[1].lower() in allowed
    )


def normalize_depth(arr: np.ndarray, invert: bool = False) -> np.ndarray:
    arr = arr.astype(np.float64)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo > 1e-6:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    if invert:
        arr = 1.0 - arr
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def save_viz(rgb: np.ndarray, depth_u8: np.ndarray, path: str) -> None:
    cmap = plt.cm.inferno(depth_u8 / 255.0)[:, :, :3]
    panel = np.hstack([rgb / 255.0, cmap])
    plt.imsave(path, panel)


def run_hf_model(image_paths: list[str], model_name: str, outputs_root: str) -> str:
    from transformers import pipeline as hf_pipeline

    slug = slugify(model_name)
    pred_dir = os.path.join(outputs_root, slug, "pred_depth")
    viz_dir = os.path.join(outputs_root, slug, "viz")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    csv_path = os.path.join(outputs_root, slug, "latency.csv")

    invert = model_name in INVERT_MODELS
    pipe = hf_pipeline(task="depth-estimation", model=model_name)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "image", "latency_seconds"])
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            t0 = time.perf_counter()
            pred = np.array(pipe(img)["depth"])
            lat = time.perf_counter() - t0

            base = os.path.splitext(os.path.basename(p))[0]
            depth_u8 = normalize_depth(pred, invert=invert)
            Image.fromarray(depth_u8).save(os.path.join(pred_dir, f"{base}_pred.png"))
            save_viz(np.array(img), depth_u8, os.path.join(viz_dir, f"{base}_panel.png"))
            w.writerow([model_name, p, f"{lat:.6f}"])

    print(f"[{model_name}] done -> {pred_dir}")
    return csv_path


def run_da3_model(image_paths: list[str], model_name: str, outputs_root: str) -> str:
    from depth_anything_3.api import DepthAnything3  # type: ignore
    import torch

    slug = slugify(model_name)
    pred_dir = os.path.join(outputs_root, slug, "pred_depth")
    viz_dir = os.path.join(outputs_root, slug, "viz")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    csv_path = os.path.join(outputs_root, slug, "latency.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_name).to(device=device)

    t0 = time.perf_counter()
    prediction = model.inference(image_paths)
    total_lat = time.perf_counter() - t0
    per_image_lat = total_lat / len(image_paths)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "image", "latency_seconds"])
        for i, p in enumerate(image_paths):
            img = Image.open(p).convert("RGB")
            pred_np = prediction.depth[i]
            if pred_np.shape[:2] != (img.height, img.width):
                pred_np = np.array(
                    Image.fromarray(pred_np.astype(np.float32), mode="F").resize(
                        (img.width, img.height), resample=Image.BILINEAR
                    )
                )
            base = os.path.splitext(os.path.basename(p))[0]
            depth_u8 = normalize_depth(pred_np, invert=True)
            Image.fromarray(depth_u8).save(os.path.join(pred_dir, f"{base}_pred.png"))
            save_viz(np.array(img), depth_u8, os.path.join(viz_dir, f"{base}_panel.png"))
            w.writerow([model_name, p, f"{per_image_lat:.6f}"])

    print(f"[{model_name}] done -> {pred_dir}")
    return csv_path


def run_tao(image_paths: list[str], outputs_root: str) -> str:
    model_name = "NVIDIA-TAO/NvDepthAnythingV2-Large"
    slug = slugify(model_name)
    pred_dir = os.path.join(outputs_root, slug, "pred_depth")
    viz_dir = os.path.join(outputs_root, slug, "viz")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    csv_path = os.path.join(outputs_root, slug, "latency.csv")

    ckpt = os.path.join(
        PROJECT_ROOT,
        "checkpoints",
        "nvdepthanythingv2_vtrainable_relative_depthanythingv2_large_v1.0",
        "trainable_relative_depthanythingv2_large_v1.0.pth",
    )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"TAO checkpoint not found: {ckpt}")

    filelist = os.path.join(PROJECT_ROOT, "specs", "sample5_mono_filelist.txt")
    os.makedirs(os.path.dirname(filelist), exist_ok=True)
    with open(filelist, "w") as fl:
        for p in image_paths:
            fl.write(p + "\n")

    tao_results = os.path.join(outputs_root, slug, "tao_raw")
    os.makedirs(tao_results, exist_ok=True)

    spec = os.path.join(PROJECT_ROOT, "specs", "experiment.yaml")
    cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-v", "/home/chamika2:/home/chamika2",
        "nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt",
        "depth_net", "inference",
        "-e", spec,
        "dataset.dataset_name=MonoDataset",
        "model.model_type=RelativeDepthAnything",
        f"inference.results_dir={tao_results}",
        f"inference.checkpoint={ckpt}",
        f"dataset.infer_dataset.data_sources=[{{data_file:{filelist},dataset_name:relativemonodataset}}]",
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_lat = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"[TAO] FAILED:\n{result.stderr[-2000:]}")
        raise RuntimeError("TAO inference failed")

    per_image_lat = total_lat / len(image_paths)

    tao_img_root = os.path.join(tao_results, "inference_images")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "image", "latency_seconds"])
        for p in image_paths:
            base = os.path.splitext(os.path.basename(p))[0]
            tao_pred_path = os.path.join(tao_img_root, p.lstrip("/"))
            if os.path.isfile(tao_pred_path):
                pred_img = Image.open(tao_pred_path).convert("L")
                pred_np = np.array(pred_img).astype(np.float32)
            else:
                print(f"  [TAO] WARNING: no output for {base}")
                pred_np = np.zeros((600, 800), dtype=np.float32)

            img = Image.open(p).convert("RGB")
            if pred_np.shape[:2] != (img.height, img.width):
                pred_np = np.array(
                    Image.fromarray(pred_np, mode="F").resize(
                        (img.width, img.height), resample=Image.BILINEAR
                    )
                )

            depth_u8 = normalize_depth(pred_np, invert=False)
            Image.fromarray(depth_u8).save(os.path.join(pred_dir, f"{base}_pred.png"))
            save_viz(np.array(img), depth_u8, os.path.join(viz_dir, f"{base}_panel.png"))
            w.writerow([model_name, p, f"{per_image_lat:.6f}"])

    print(f"[TAO] done -> {pred_dir}")
    return csv_path


def summarize(csv_paths: Iterable[str], outputs_root: str) -> None:
    summary_path = os.path.join(outputs_root, "latency_summary.csv")
    rows = []
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            vals = [float(r["latency_seconds"]) for r in csv.DictReader(f)]
        if not vals:
            continue
        model = os.path.basename(os.path.dirname(path)).replace("__", "/")
        rows.append({
            "model": model,
            "num_images": len(vals),
            "mean_s": f"{np.mean(vals):.4f}",
            "min_s": f"{np.min(vals):.4f}",
            "max_s": f"{np.max(vals):.4f}",
        })

    fields = ["model", "num_images", "mean_s", "min_s", "max_s"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["hf_baselines", "da3", "tao"],
        required=True,
    )
    parser.add_argument("--input-dir", default=IMAGES_DIR)
    args = parser.parse_args()

    image_paths = collect_images(args.input_dir)
    if not image_paths:
        raise ValueError(f"No images found in {args.input_dir}")

    outputs_root = os.path.join(PROJECT_ROOT, "outputs", args.mode)
    os.makedirs(outputs_root, exist_ok=True)

    csv_paths: list[str] = []

    if args.mode == "hf_baselines":
        for m in HF_BASELINES:
            try:
                csv_paths.append(run_hf_model(image_paths, m, outputs_root))
            except Exception as e:
                print(f"[{m}] SKIPPED: {e}")

    elif args.mode == "da3":
        for m in DA3_MODELS:
            try:
                csv_paths.append(run_da3_model(image_paths, m, outputs_root))
            except Exception as e:
                print(f"[{m}] SKIPPED: {e}")

    elif args.mode == "tao":
        try:
            csv_paths.append(run_tao(image_paths, outputs_root))
        except Exception as e:
            print(f"[TAO] SKIPPED: {e}")

    summarize(csv_paths, outputs_root)


if __name__ == "__main__":
    main()
