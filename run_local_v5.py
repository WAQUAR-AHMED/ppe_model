"""
run_local_v5.py - Local runner using a YOLO pose model for person/keypoints
and a PPE detection model for equipment checks.

Behavior:
  - Pose estimation -> head/torso/feet regions.
  - PPE detections are evaluated within those regions.
  - Inference runs on every frame (no skipping).
"""

import sys
import os
import argparse
import time
import cv2
import torch
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PPE_CODE_DIR = os.path.join(SCRIPT_DIR, "src", "local_models", "ppe_code")

DEFAULT_MODEL_PATH = os.path.join(PPE_CODE_DIR, "best_m.pt")
FALLBACK_MODEL_PATH = os.path.join(PPE_CODE_DIR, "bet_m.pt")
DEFAULT_POSE_MODEL_PATH = os.path.join(PPE_CODE_DIR, "yolo11m-pose.pt")

sys.path.insert(0, PPE_CODE_DIR)
from ppe_logic_v5 import PPELogicV5  # noqa: E402
from ultralytics import YOLO  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on: {DEVICE}")

PERSON_COLOR = (0, 165, 255)
POSE_CONF_THRESH = 0.70


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPE Detection v5 - pose-driven person regions with PPE checks"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: '0' for webcam, or a path to a video file (default: 0)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to YOLO .pt model file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--pose-model",
        default=DEFAULT_POSE_MODEL_PATH,
        help=f"Path to YOLO pose .pt model file (default: {DEFAULT_POSE_MODEL_PATH})",
    )
    parser.add_argument(
        "--required",
        default=None,
        help=(
            "Comma-separated list of PPE items to enforce. "
            "Leave blank to enforce all PPE classes. "
            "Example: --required 'helmet,vest,boots'"
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Detection confidence threshold for tracking (default: 0.40)",
    )
    parser.add_argument(
        "--process-every-n",
        type=int,
        default=1,
        help="Run YOLO inference on every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the displayed output to annotated_output_v5.mp4",
    )
    return parser.parse_args()


def _resolve_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    if model_path == DEFAULT_MODEL_PATH and os.path.exists(FALLBACK_MODEL_PATH):
        print(f"[WARN] Default model not found at {DEFAULT_MODEL_PATH}")
        print(f"[WARN] Falling back to: {FALLBACK_MODEL_PATH}")
        return FALLBACK_MODEL_PATH
    raise FileNotFoundError(
        f"[ERROR] Model not found at:\n  {model_path}\n"
        "Make sure the .pt file exists in src/local_models/ppe_code/ or pass --model <path>"
    )


def _resolve_pose_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    raise FileNotFoundError(
        f"[ERROR] Pose model not found at:\n  {model_path}\n"
        "Make sure the .pt file exists in src/local_models/ppe_code/ or pass --pose-model <path>"
    )


def _scale_gap_param(value: int, process_every_n: int) -> int:
    return max(1, int(value) * max(1, int(process_every_n)))


def load_model(model_path: str, process_every_n: int, required_ppe=None):
    resolved_model_path = _resolve_model_path(model_path)

    print(f"[INFO] Loading model from: {resolved_model_path}")
    model = YOLO(resolved_model_path)
    model.to(DEVICE)

    print(f"[INFO] Model class names: {model.names}")

    model.ppe_logic = PPELogicV5(
        model_class_names=model.names,
        required_ppe=required_ppe,
        class_conf_thresholds={
            "person": 0.50,
            "helmet": 0.55,
            "vest": 0.50,
            "boots": 0.45,
        },
        grace_frames=_scale_gap_param(90, process_every_n),
        confirm_frames=_scale_gap_param(5, process_every_n),
        resolve_frames=_scale_gap_param(3, process_every_n),
        state_ttl_frames=_scale_gap_param(120, process_every_n),
        emit_violation_updates=True,
        alert_cooldown_seconds=900,
        equipment_hold_frames=_scale_gap_param(900, process_every_n),
        reid_max_gap_frames=_scale_gap_param(30, process_every_n),
        reid_min_iou=0.20,
        reid_max_center_distance=60.0 * max(1, int(process_every_n)),
        overlap_threshold=0.0,
    )

    enforced = [
        model.names[cid]
        for cid in sorted(model.ppe_logic.required_ppe_ids)
    ]
    print(
        f"[INFO] Person class ID  : {model.ppe_logic.person_class_id} "
        f"({model.names[model.ppe_logic.person_class_id]})"
    )
    print(f"[INFO] Enforced PPE     : {enforced}")
    print(f"[INFO] Process every N : {process_every_n}")
    print("[INFO] Model loaded successfully.")
    return model


def load_pose_model(model_path: str):
    resolved_model_path = _resolve_pose_model_path(model_path)
    print(f"[INFO] Loading pose model from: {resolved_model_path}")
    model = YOLO(resolved_model_path)
    model.to(DEVICE)
    print("[INFO] Pose model loaded successfully.")
    return model


def _bbox_from_points(points, pad=8):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not xs or not ys:
        return None
    x1 = int(max(0, min(xs) - pad))
    y1 = int(max(0, min(ys) - pad))
    x2 = int(max(xs) + pad)
    y2 = int(max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _build_pose_regions(pose_result):
    persons = []

    if pose_result.keypoints is None or pose_result.boxes is None:
        return persons

    kpts_xy = pose_result.keypoints.xy
    kpts_conf = pose_result.keypoints.conf
    boxes = pose_result.boxes

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        raw_pid = int(box.id.item()) if box.id is not None else -1
        conf = float(box.conf.item())

        kp_xy = kpts_xy[idx]
        kp_cf = kpts_conf[idx] if kpts_conf is not None else None

        def _pt(i, min_conf=0.2):
            if kp_xy is None:
                return None
            x, y = float(kp_xy[i][0]), float(kp_xy[i][1])
            if kp_cf is not None and float(kp_cf[i]) < min_conf:
                return None
            if x <= 0 and y <= 0:
                return None
            return (x, y)

        # COCO-17 indices
        head_pts = [p for p in [
            _pt(0), _pt(1), _pt(2), _pt(3), _pt(4)
        ] if p is not None]
        torso_pts = [p for p in [
            _pt(5), _pt(6), _pt(11), _pt(12)
        ] if p is not None]
        feet_pts = [p for p in [
            _pt(15), _pt(16)
        ] if p is not None]

        head_box = _bbox_from_points(head_pts, pad=12) if head_pts else None
        if head_box is not None:
            hx1, hy1, hx2, hy2 = head_box
            # Extend head region to the top of the person bbox
            head_box = (hx1, y1, hx2, hy2)
        torso_box = _bbox_from_points(torso_pts, pad=14) if torso_pts else None
        feet_box = _bbox_from_points(feet_pts, pad=10) if feet_pts else None

        persons.append({
            "raw_pid": raw_pid,
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
            "regions": {
                "head": head_box,
                "torso": torso_box,
                "feet": feet_box,
            },
        })

    return persons


class SkipFrameTracker:
    def __init__(self, ppe_logic, frame_w: int, frame_h: int):
        self.ppe_logic = ppe_logic
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.track_memory = {}

    def update_from_inference(self, detections, frame_num: int):
        seen_ids = set()
        for det in detections:
            sid = int(det["person_id"])
            bbox = [int(v) for v in det["bbox"]]
            prev = self.track_memory.get(sid)
            velocity = [0, 0, 0, 0]

            if prev is not None:
                gap = max(1, frame_num - prev["frame_num"])
                velocity = [
                    (bbox[idx] - prev["bbox"][idx]) / float(gap)
                    for idx in range(4)
                ]

            self.track_memory[sid] = {
                "bbox": bbox,
                "velocity": velocity,
                "frame_num": frame_num,
                "ppe_status": dict(det.get("ppe_status", {})),
            }
            seen_ids.add(sid)

        stale_ids = [
            sid for sid, memory in self.track_memory.items()
            if frame_num - memory["frame_num"] > self.ppe_logic.state_ttl_frames
        ]
        for sid in stale_ids:
            self.track_memory.pop(sid, None)

    def predict_to_frame(self, frame_num: int):
        predicted = []
        for sid, memory in list(self.track_memory.items()):
            steps = frame_num - memory["frame_num"]
            if steps <= 0:
                predicted.append((sid, list(memory["bbox"]), memory))
                continue

            bbox = []
            for idx, value in enumerate(memory["bbox"]):
                projected = int(round(value + memory["velocity"][idx] * steps))
                bbox.append(projected)

            bbox[0] = max(0, min(self.frame_w - 1, bbox[0]))
            bbox[2] = max(0, min(self.frame_w - 1, bbox[2]))
            bbox[1] = max(0, min(self.frame_h - 1, bbox[1]))
            bbox[3] = max(0, min(self.frame_h - 1, bbox[3]))

            if bbox[2] <= bbox[0]:
                bbox[2] = min(self.frame_w - 1, bbox[0] + 1)
            if bbox[3] <= bbox[1]:
                bbox[3] = min(self.frame_h - 1, bbox[1] + 1)

            predicted.append((sid, bbox, memory))
        return predicted

    def sync_logic_state(self, frame_num: int):
        for sid, bbox, memory in self.predict_to_frame(frame_num):
            self.ppe_logic.stable_tracks[sid] = {
                "bbox": tuple(bbox),
                "last_seen_frame": frame_num,
            }
            if sid in self.ppe_logic.person_states:
                self.ppe_logic.person_states[sid]["last_seen_frame"] = frame_num

            memory["bbox"] = bbox
            memory["frame_num"] = frame_num

    def render_predicted(self, frame, frame_num: int):
        output = frame.copy()
        for sid, bbox, memory in self.predict_to_frame(frame_num):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), PERSON_COLOR, 2)
            cv2.putText(
                output,
                f"SID:{sid} PRED",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                PERSON_COLOR,
                2,
            )
            summary = " ".join(
                f"{name[:3].upper()}:{'yes' if status == 'yes' else ('NA' if status == 'na' else 'No')}"
                for name, status in memory.get("ppe_status", {}).items()
            )
            if summary:
                cv2.putText(
                    output,
                    summary,
                    (x1, min(self.frame_h - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
        return output


def _draw_timestamp(frame, frame_num: int, infer_ran: bool):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    mode = "INFER" if infer_ran else "SKIP"
    cv2.putText(
        frame,
        f"{ts} | Frame {frame_num} | {mode}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if infer_ran else (0, 255, 255),
        2,
    )
    return frame


def run(
    source,
    model_path,
    pose_model_path,
    conf_thresh,
    process_every_n=1,
    skip_frame_mode="raw",
    save=False,
    required_ppe=None,
):
    if process_every_n < 1:
        raise ValueError("--process-every-n must be >= 1")

    # Force no frame skipping for v5.
    process_every_n = 1

    model = load_model(model_path, process_every_n=process_every_n, required_ppe=required_ppe)
    pose_model = load_pose_model(pose_model_path)
    ppe_logic = model.ppe_logic

    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Could not open video source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    print(f"[INFO] Source: {source}  |  Resolution: {frame_w}x{frame_h}  |  FPS: {fps:.1f}")

    tracker_cfg = os.path.join(PPE_CODE_DIR, "bytetrack_custom.yaml")

    writer = None
    if save:
        out_path = os.path.join(SCRIPT_DIR, "annotated_output_v5.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))
        print(f"[INFO] Saving output to: {out_path}")

    frame_counter = 0
    inferred_frames = 0
    print("[INFO] Starting inference... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream / no frame.")
            break

        frame_counter += 1
        inferred_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        pose_results = pose_model.track(
            source=image_pil,
            conf=POSE_CONF_THRESH,
            tracker=tracker_cfg,
            persist=True,
            stream=False,
            verbose=False,
            device=DEVICE,
        )

        persons = _build_pose_regions(pose_results[0])

        ppe_items = []
        for person in persons:
            px1, py1, px2, py2 = person["bbox"]
            px1 = max(0, px1)
            py1 = max(0, py1)
            px2 = min(frame_w - 1, px2)
            py2 = min(frame_h - 1, py2)
            if px2 <= px1 or py2 <= py1:
                continue

            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            ppe_results = model.predict(
                source=crop_pil,
                conf=conf_thresh,
                stream=False,
                verbose=False,
                device=DEVICE,
            )

            if not ppe_results:
                continue

            for box in ppe_results[0].boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id not in ppe_logic.ppe_class_map:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                fx1 = int(x1 + px1)
                fy1 = int(y1 + py1)
                fx2 = int(x2 + px1)
                fy2 = int(y2 + py1)
                ppe_items.append((cls_id, (fx1, fy1, fx2, fy2), conf))

        output_frame, detections, alerts = ppe_logic.process_frame_pose_items(
            frame, ppe_items, model.names, persons, frame_num=frame_counter
        )

        if alerts:
            for alert in alerts:
                missing_items = [k for k, v in alert["ppe_status"].items() if v == "no"]
                print(
                    f"[ALERT] Frame {frame_counter} | Person SID {alert['person_id']} "
                    f"| Status: {alert['status']} | Missing: {missing_items}"
                )

        output_frame = _draw_timestamp(output_frame, frame_counter, True)
        cv2.imshow("PPE Detection v5 - press Q to quit", output_frame)

        if writer is not None:
            writer.write(output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit requested.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(
        f"[INFO] Done. Processed {frame_counter} frames, "
        f"ran inference on {inferred_frames} frames."
    )


if __name__ == "__main__":
    args = parse_args()

    required_ppe = None
    if args.required:
        required_ppe = [item.strip() for item in args.required.split(",") if item.strip()]
        print(f"[INFO] Enforcing PPE subset: {required_ppe}")

    run(
        source=args.source,
        model_path=args.model,
        pose_model_path=args.pose_model,
        conf_thresh=args.conf,
        process_every_n=args.process_every_n,
        skip_frame_mode="raw",
        save=args.save,
        required_ppe=required_ppe,
    )
