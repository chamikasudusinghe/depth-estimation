import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def yuv420_888_to_bgr(raw: np.ndarray, width: int, height: int, planes: list) -> np.ndarray:
    row_stride_y    = planes[0]["rowStride"]
    y_buf_size      = planes[0]["bufferSize"]
    row_stride_uv   = planes[1]["rowStride"]
    pixel_stride_uv = planes[1]["pixelStride"]

    y_plane = np.empty((height, width), dtype=np.uint8)
    for row in range(height):
        s = row * row_stride_y
        y_plane[row, :] = raw[s:s + width]

    chroma_w = width  // 2
    chroma_h = height // 2
    u_plane  = np.empty((chroma_h, chroma_w), dtype=np.uint8)
    v_plane  = np.empty((chroma_h, chroma_w), dtype=np.uint8)

    if pixel_stride_uv == 1:
        uv_buf = planes[1]["bufferSize"]
        for row in range(chroma_h):
            s = y_buf_size + row * row_stride_uv
            u_plane[row, :] = raw[s:s + chroma_w]
        for row in range(chroma_h):
            s = y_buf_size + uv_buf + row * planes[2]["rowStride"]
            v_plane[row, :] = raw[s:s + chroma_w]
    else:
        for row in range(chroma_h):
            s = y_buf_size + row * row_stride_uv
            d = raw[s:s + chroma_w * pixel_stride_uv]
            u_plane[row, :] = d[0::2][:chroma_w]
            v_plane[row, :] = d[1::2][:chroma_w]

    i420 = np.concatenate([y_plane.ravel(), u_plane.ravel(), v_plane.ravel()])
    return cv2.cvtColor(i420.reshape((height * 3) // 2, width), cv2.COLOR_YUV2BGR_I420)


def ndc_to_linear_depth(depth_ndc: np.ndarray, near: float, far: float) -> np.ndarray:
    if np.isinf(far) or far < near:
        x, y = -2.0 * near, -1.0
    else:
        x = -2.0 * far * near / (far - near)
        y = -(far + near) / (far - near)
    ndc   = depth_ndc * 2.0 - 1.0
    denom = ndc + y
    return np.divide(x, denom, out=np.zeros_like(depth_ndc), where=denom != 0).astype(np.float32)


def depth_intrinsics_from_fov(left, right, top, bottom, width, height):
    fx = width  / (right + left)
    fy = height / (top   + bottom)
    cx = width  * right / (right + left)
    cy = height * top   / (top   + bottom)
    return fx, fy, cx, cy


def load_camera_local_transform(characteristics_json: Path):
    with open(characteristics_json) as f:
        cc = json.load(f)
    pose   = cc["pose"]
    transl = list(pose["translation"])
    transl[2] *= -1
    q   = pose["rotation"]
    rot = R.from_quat((-q[0], -q[1], q[2], q[3])).inv()
    rot = rot * R.from_euler("x", np.pi)
    return np.array(transl, dtype=np.float64), rot.as_quat()


def apply_local_transform(hmd_pos, hmd_quat, local_pos, local_quat):
    parent  = R.from_quat(hmd_quat)
    cam_pos = hmd_pos + parent.apply(local_pos)
    cam_rot = (parent * R.from_quat(local_quat)).as_quat()
    return cam_pos, cam_rot


class PoseInterpolator:
    def __init__(self, poses_csv: Path):
        rows = []
        with open(poses_csv) as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({
                        "t":   int(row["unix_time"]),
                        "pos": np.array([float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])]),
                        "rot": np.array([float(row["rot_x"]), float(row["rot_y"]), float(row["rot_z"]), float(row["rot_w"])]),
                    })
                except (ValueError, KeyError):
                    continue
        self._rows = sorted(rows, key=lambda r: r["t"])
        self._ts   = np.array([r["t"] for r in self._rows])

    def interpolate(self, timestamp: int, window: int = 30):
        idx = np.searchsorted(self._ts, timestamp)
        lo  = idx - 1 if idx > 0              else None
        hi  = idx     if idx < len(self._rows) else None
        if lo is None or hi is None:
            return None
        if abs(self._rows[lo]["t"] - timestamp) > window:
            return None
        if abs(self._rows[hi]["t"] - timestamp) > window:
            return None
        t0, t1 = self._rows[lo]["t"], self._rows[hi]["t"]
        alpha  = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0
        pos  = (1 - alpha) * self._rows[lo]["pos"] + alpha * self._rows[hi]["pos"]
        rots = R.from_quat([self._rows[lo]["rot"], self._rows[hi]["rot"]])
        rot  = Slerp([0, 1], rots)(alpha).as_quat()
        return pos, rot


def make_c2w(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.from_quat(quat).as_matrix().astype(np.float32)
    c2w[:3, 3] = pos.astype(np.float32)
    return c2w


def nearest_ts(query: int, candidates: np.ndarray) -> int:
    return int(candidates[np.argmin(np.abs(candidates - query))])


METADATA_FIELDS = [
    "rgb_ts", "depth_ts",
    "rgb_fx", "rgb_fy", "rgb_cx", "rgb_cy", "rgb_width", "rgb_height",
    "depth_fx", "depth_fy", "depth_cx", "depth_cy", "depth_width", "depth_height",
    "near", "far",
    "rgb_pos_x", "rgb_pos_y", "rgb_pos_z",
    "rgb_rot_x", "rgb_rot_y", "rgb_rot_z", "rgb_rot_w",
    "depth_pos_x", "depth_pos_y", "depth_pos_z",
    "depth_rot_x", "depth_rot_y", "depth_rot_z", "depth_rot_w",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", "-s", type=Path, required=True)
    parser.add_argument("--output-dir",  "-o", type=Path, required=True)
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    sd, out, side = args.session_dir, args.output_dir, args.side

    yuv_dir    = sd / f"{side}_camera_raw"
    depth_dir  = sd / f"{side}_depth"
    fmt_json   = sd / f"{side}_camera_image_format.json"
    chars_json = sd / f"{side}_camera_characteristics.json"
    depth_csv  = sd / f"{side}_depth_descriptors.csv"
    hmd_csv    = sd / "hmd_poses.csv"

    for p in (yuv_dir, depth_dir, fmt_json, chars_json, depth_csv, hmd_csv):
        if not p.exists():
            raise FileNotFoundError(p)

    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "pred_depth").mkdir(parents=True, exist_ok=True)
    (out / "rgb_extrinsics").mkdir(parents=True, exist_ok=True)
    (out / "depth_extrinsics").mkdir(parents=True, exist_ok=True)

    with open(fmt_json) as f:
        fmt = json.load(f)
    width, height, planes = fmt["width"], fmt["height"], fmt["planes"]

    with open(chars_json) as f:
        cc = json.load(f)
    intr = cc["intrinsics"]
    rgb_fx, rgb_fy = intr["fx"], intr["fy"]
    rgb_cx, rgb_cy = intr["cx"], intr["cy"]

    local_pos, local_quat = load_camera_local_transform(chars_json)

    depth_rows = []
    with open(depth_csv) as f:
        for row in csv.DictReader(f):
            depth_rows.append({
                "ts":   int(row["timestamp_ms"]),
                "w":    int(row["width"]),
                "h":    int(row["height"]),
                "near": float(row["near_z"]),
                "far":  float(row["far_z"]),
                "left": float(row["fov_left_angle_tangent"]),
                "right":float(row["fov_right_angle_tangent"]),
                "top":  float(row["fov_top_angle_tangent"]),
                "bot":  float(row["fov_down_angle_tangent"]),
                "px":   float(row["create_pose_location_x"]),
                "py":   float(row["create_pose_location_y"]),
                "pz":   float(row["create_pose_location_z"]),
                "rx":   float(row["create_pose_rotation_x"]),
                "ry":   float(row["create_pose_rotation_y"]),
                "rz":   float(row["create_pose_rotation_z"]),
                "rw":   float(row["create_pose_rotation_w"]),
            })
    depth_ts_arr = np.array([r["ts"] for r in depth_rows])
    depth_meta   = {r["ts"]: r for r in depth_rows}

    interp    = PoseInterpolator(hmd_csv)
    yuv_files = sorted(yuv_dir.glob("*.yuv"), key=lambda p: int(p.stem))
    if args.max_frames > 0:
        yuv_files = yuv_files[:args.max_frames]

    print(f"RGB frames  : {len(yuv_files)}")
    print(f"Depth frames: {len(depth_rows)}")
    print(f"Output dir  : {out.resolve()}\n")

    written = skipped = no_pose = 0

    with open(out / "metadata.csv", "w", newline="") as meta_f:
        writer = csv.DictWriter(meta_f, fieldnames=METADATA_FIELDS)
        writer.writeheader()

        for yuv_path in yuv_files:
            rgb_ts    = int(yuv_path.stem)
            depth_ts  = nearest_ts(rgb_ts, depth_ts_arr)
            raw_depth = depth_dir / f"{depth_ts}.raw"

            if not raw_depth.exists():
                skipped += 1
                continue

            meta      = depth_meta[depth_ts]
            depth_ndc = np.fromfile(raw_depth, dtype="<f4").reshape((meta["h"], meta["w"]))

            if np.isnan(depth_ndc).any() or not (depth_ndc != 0).any():
                skipped += 1
                continue

            hmd_pose = interp.interpolate(rgb_ts)
            if hmd_pose is None:
                no_pose += 1
                continue
            rgb_pos, rgb_rot = apply_local_transform(*hmd_pose, local_pos, local_quat)

            dfx, dfy, dcx, dcy = depth_intrinsics_from_fov(
                meta["left"], meta["right"], meta["top"], meta["bot"],
                meta["w"], meta["h"]
            )

            raw_yuv = np.fromfile(yuv_path, dtype=np.uint8)
            cv2.imwrite(str(out / "rgb" / f"{rgb_ts}.png"),
                        yuv420_888_to_bgr(raw_yuv, width, height, planes))

            np.save(str(out / "depth" / f"{depth_ts}.npy"),
                    ndc_to_linear_depth(depth_ndc, meta["near"], meta["far"]))

            np.save(str(out / "rgb_extrinsics" / f"{rgb_ts}.npy"),
                    make_c2w(rgb_pos, rgb_rot))
            depth_quat = np.array([meta["rx"], meta["ry"], meta["rz"], meta["rw"]])
            depth_pos  = np.array([meta["px"], meta["py"], meta["pz"]])
            np.save(str(out / "depth_extrinsics" / f"{depth_ts}.npy"),
                    make_c2w(depth_pos, depth_quat))

            writer.writerow({
                "rgb_ts": rgb_ts, "depth_ts": depth_ts,
                "rgb_fx": rgb_fx, "rgb_fy": rgb_fy,
                "rgb_cx": rgb_cx, "rgb_cy": rgb_cy,
                "rgb_width": width, "rgb_height": height,
                "depth_fx": dfx, "depth_fy": dfy,
                "depth_cx": dcx, "depth_cy": dcy,
                "depth_width": meta["w"], "depth_height": meta["h"],
                "near": meta["near"], "far": meta["far"],
                "rgb_pos_x": rgb_pos[0], "rgb_pos_y": rgb_pos[1], "rgb_pos_z": rgb_pos[2],
                "rgb_rot_x": rgb_rot[0], "rgb_rot_y": rgb_rot[1],
                "rgb_rot_z": rgb_rot[2], "rgb_rot_w": rgb_rot[3],
                "depth_pos_x": meta["px"], "depth_pos_y": meta["py"], "depth_pos_z": meta["pz"],
                "depth_rot_x": meta["rx"], "depth_rot_y": meta["ry"],
                "depth_rot_z": meta["rz"], "depth_rot_w": meta["rw"],
            })

            written += 1
            if written % 20 == 1:
                print(f"  [{written:3d}] rgb={rgb_ts}  depth={depth_ts}")

    print(f"\nWrote {written} frames  |  skipped {skipped}  |  no-pose {no_pose}")


if __name__ == "__main__":
    main()
