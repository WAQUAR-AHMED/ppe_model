"""
run_local.py  –  Standalone local runner for best.pt PPE detection model.

Usage:
    # Webcam (default)
    python run_local.py

    # Local video file
    python run_local.py --source "C:\\path\\to\\video.mp4"

Press 'q' to quit the live window.
"""

import sys
import os
import argparse
import cv2
import torch
import numpy as np
from PIL import Image

# ── Resolve paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PPE_CODE_DIR = os.path.join(SCRIPT_DIR, "src", "local_models", "ppe_code")
MODEL_PATH  = os.path.join(PPE_CODE_DIR, "best.pt")

# Make ppe_logic importable without any package magic
sys.path.insert(0, PPE_CODE_DIR)
from ppe_logic import PPELogic  # noqa: E402  (imported after path fix)
from ultralytics import YOLO   # noqa: E402

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on: {DEVICE}")

# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="PPE Detection – local runner")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: '0' for webcam, or a path to a video file (default: 0)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated output to annotated_output.mp4",
    )
    return parser.parse_args()


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"[ERROR] best.pt not found at:\n  {MODEL_PATH}\n"
            "Make sure the file is inside src/local_models/ppe_code/"
        )
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    model.ppe_logic = PPELogic(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
    return model


# ── Inference loop ─────────────────────────────────────────────────────────────
def run(source, save=False):
    model    = load_model()
    ppe_logic = model.ppe_logic

    # Support integer index for webcam
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Could not open video source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 20.0
    print(f"[INFO] Source: {source}  |  Resolution: {frame_w}x{frame_h}  |  FPS: {fps:.1f}")

    # Use custom tracker config (lives next to this script)
    tracker_cfg = os.path.join(SCRIPT_DIR, "bytetrack_custom.yaml")

    writer = None
    if save:
        out_path = os.path.join(SCRIPT_DIR, "annotated_output.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))
        print(f"[INFO] Saving output to: {out_path}")

    frame_counter = 0
    print("[INFO] Starting inference… Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream / no frame.")
            break

        frame_counter += 1

        # Convert BGR → RGB PIL image for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        # Track
        results = model.track(
            source=image_pil,
            conf=0.45,
            tracker=tracker_cfg,
            persist=True,
            stream=False,
            verbose=False,
            device=DEVICE,
        )

        # PPE logic → annotated frame
        annotated_frame, detections, alerts = ppe_logic.process_frame(
            results[0], frame_num=frame_counter
        )

        # Print alerts to console
        if alerts:
            for alert in alerts:
                print(f"[ALERT] Frame {frame_counter} | Person ID {alert['person_id']} "
                      f"| PPE: {alert['ppe_status']}")

        # Show window
        cv2.imshow("PPE Detection  –  press Q to quit", annotated_frame)

        if writer is not None:
            writer.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit requested.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_counter} frames.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run(source=args.source, save=args.save)

