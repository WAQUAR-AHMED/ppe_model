"""
run_local_v9.py - Local runner for best_new.pt with yolo11m-pose regions.
"""

import argparse
import os
import sys
import time

import cv2
import torch
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PPE_CODE_DIR = os.path.join(SCRIPT_DIR, "src", "local_models", "ppe_code")

DEFAULT_MODEL_PATH = os.path.join(PPE_CODE_DIR, "best_new.pt")
FALLBACK_MODEL_PATH = os.path.join(PPE_CODE_DIR, "best_m.pt")
DEFAULT_POSE_MODEL_PATH = os.path.join(PPE_CODE_DIR, "yolo11m-pose.pt")

sys.path.insert(0, PPE_CODE_DIR)
from pose_regions import propose_foot_region, propose_head_region, propose_torso_region  # noqa: E402
from ppe_logic_v9 import PPELogicV9  # noqa: E402
from ultralytics import YOLO  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on: {DEVICE}")
POSE_CONF_THRESH = 0.70


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPE Detection v9 - best_new.pt + yolo11m-pose"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: '0' for webcam, or a path to a video file (default: 0)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to PPE YOLO .pt model file (default: {DEFAULT_MODEL_PATH})",
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
            "Comma-separated PPE list to enforce. "
            "Example: --required 'helmet,mask,safety_vest,boots,glasses,glove'"
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="PPE detection confidence threshold (default: 0.40)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output to annotated_output_v9.mp4",
    )
    parser.add_argument(
        "--show-regions",
        action="store_true",
        help="Draw pose regions (head/torso/feet/hands) on output",
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


def load_model(model_path: str, required_ppe=None, show_regions=False):
    resolved_model_path = _resolve_model_path(model_path)
    print(f"[INFO] Loading PPE model from: {resolved_model_path}")

    model = YOLO(resolved_model_path)
    model.to(DEVICE)
    print(f"[INFO] Model class names: {model.names}")

    model.ppe_logic = PPELogicV9(
        model_class_names=model.names,
        required_ppe=required_ppe,
        show_regions=show_regions,
        class_conf_thresholds={
            "person": 0.50,
            "helmet": 0.55,
            "mask": 0.15,
            "safety_vest": 0.50,
            "boots": 0.45,
            "glasses": 0.35,
            "glove": 0.65,
        },
        grace_frames=90,
        confirm_frames=5,
        resolve_frames=3,
        state_ttl_frames=120,
        emit_violation_updates=True,
        alert_cooldown_seconds=900,
        equipment_hold_frames=900,
        reid_max_gap_frames=30,
        reid_min_iou=0.20,
        reid_max_center_distance=60.0,
        overlap_threshold=0.0,
    )

    enforced = [model.names[cid] for cid in sorted(model.ppe_logic.required_ppe_ids)]
    print(
        f"[INFO] Person class ID  : {model.ppe_logic.person_class_id} "
        f"({model.names[model.ppe_logic.person_class_id]})"
    )
    print(f"[INFO] Enforced PPE     : {enforced}")
    print("[INFO] PPE model loaded successfully.")
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


def _merge_boxes(boxes):
    valid = [box for box in boxes if box is not None]
    if not valid:
        return None
    x1 = min(box[0] for box in valid)
    y1 = min(box[1] for box in valid)
    x2 = max(box[2] for box in valid)
    y2 = max(box[3] for box in valid)
    if x2 <= x1 or y2 <= y1:
        return None
    return (int(x1), int(y1), int(x2), int(y2))


def _clip_box_to_bbox(box, bounds):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    bx1, by1, bx2, by2 = bounds
    x1 = max(bx1, x1)
    y1 = max(by1, y1)
    x2 = min(bx2, x2)
    y2 = min(by2, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (int(x1), int(y1), int(x2), int(y2))


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

        person_bbox = (x1, y1, x2, y2)
        head_box = propose_head_region(kp_xy, kp_cf, person_bbox, conf_threshold=0.2)
        torso_box = propose_torso_region(kp_xy, kp_cf, person_bbox, conf_threshold=0.2)
        feet_box = propose_foot_region(kp_xy, kp_cf, person_bbox, conf_threshold=0.2)

        left_hand_pts = [p for p in [_pt(7), _pt(9)] if p is not None]
        right_hand_pts = [p for p in [_pt(8), _pt(10)] if p is not None]
        left_hand_box = _bbox_from_points(left_hand_pts, pad=20) if left_hand_pts else None
        right_hand_box = _bbox_from_points(right_hand_pts, pad=20) if right_hand_pts else None
        left_hand_box = _clip_box_to_bbox(left_hand_box, person_bbox)
        right_hand_box = _clip_box_to_bbox(right_hand_box, person_bbox)
        hands_box = _merge_boxes([left_hand_box, right_hand_box])
        hands_box = _clip_box_to_bbox(hands_box, person_bbox)

        persons.append({
            "raw_pid": raw_pid,
            "bbox": person_bbox,
            "conf": conf,
            "regions": {
                "head": head_box,
                "torso": torso_box,
                "feet": feet_box,
                "hands": hands_box,
            },
        })

    return persons


def run(source, model_path, pose_model_path, conf_thresh, save=False, required_ppe=None, show_regions=False):
    model = load_model(model_path, required_ppe=required_ppe, show_regions=show_regions)
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

    tracker_cfg = os.path.join(SCRIPT_DIR, "bytetrack_custom.yaml")

    writer = None
    if save:
        out_path = os.path.join(SCRIPT_DIR, "annotated_output_v9.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))
        print(f"[INFO] Saving output to: {out_path}")

    frame_counter = 0
    prev_ts = time.perf_counter()
    fps_ema = 0.0
    print("[INFO] Starting inference... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream / no frame.")
            break

        frame_counter += 1
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
                ppe_items.append((
                    cls_id,
                    (int(x1 + px1), int(y1 + py1), int(x2 + px1), int(y2 + py1)),
                    conf,
                ))

        output_frame, detections, alerts = ppe_logic.process_frame_pose_items(
            frame, ppe_items, model.names, persons, frame_num=frame_counter
        )

        now_ts = time.perf_counter()
        dt = max(1e-6, now_ts - prev_ts)
        instant_fps = 1.0 / dt
        fps_ema = instant_fps if fps_ema <= 0.0 else (0.9 * fps_ema + 0.1 * instant_fps)
        prev_ts = now_ts
        cv2.putText(
            output_frame,
            f"FPS: {fps_ema:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        if alerts:
            for alert in alerts:
                missing_items = [k for k, v in alert["ppe_status"].items() if v == "no"]
                print(
                    f"[ALERT] Frame {frame_counter} | Person SID {alert['person_id']} "
                    f"| Status: {alert['status']} | Missing: {missing_items}"
                )

        cv2.imshow("PPE Detection v9 - press Q to quit", output_frame)

        if writer is not None:
            writer.write(output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit requested.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_counter} frames.")


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
        save=args.save,
        required_ppe=required_ppe,
        show_regions=args.show_regions,
    )
