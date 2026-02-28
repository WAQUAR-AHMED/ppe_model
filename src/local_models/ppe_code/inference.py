import os
import io
import json
import base64
import torch
import cv2
from PIL import Image
from ultralytics import YOLO


import sys

# # Add <project_root>/src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TRACKER_CFG = os.path.join(PROJECT_ROOT, "bytetrack_custom.yaml")


from .ppe_logic import PPELogic

FRAME_WARMUP_RUNS = 3
REQUIREMENTS_PATH = "/opt/ml/model/code/requirements.txt"
MODEL_ENV_NAME = os.environ.get("SAGEMAKER_MODEL_NAME", "ppe_model")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Inference will run on: {DEVICE}")


# ---------- Load model ----------
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model = YOLO(model_path).to(DEVICE)
    model.eval()

    # Warmup
    if DEVICE.type == "cuda":
        for _ in range(FRAME_WARMUP_RUNS):
            dummy = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=DEVICE)
            _ = model(dummy, verbose=False, device=DEVICE)

    # Attach PPELogic
    model.ppe_logic = PPELogic(model_path)
    return model


# ---------- Input parser ----------
def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    data = json.loads(request_body)
    if "image" not in data:
        raise ValueError("JSON must contain 'image' field with base64 string")

    image_bytes = base64.b64decode(data["image"])
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ---------- Prediction ----------
frame_counter = 0  # global frame counter

def predict_fn(input_data, model):
    global frame_counter
    frame_counter += 1

    # Use YOLO tracking for persistent IDs
    results = model.track(
        source=input_data,
        conf=0.1,
        tracker=TRACKER_CFG,
        batch=32,
        persist=True,
        stream=False,
        verbose=False,
        device=DEVICE
    )

    # Apply PPE logic to get annotated frame + person info
    frame, detections_json,alert= model.ppe_logic.process_frame(results[0], frame_num=frame_counter)

    # Encode annotated frame as base64
    _, buffer = cv2.imencode(".jpg", frame)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    # Wrap output by frame
    output = {
        "frame": frame_counter,
        "annotated_frame": annotated_b64,
        "detections": detections_json,
        "alerts":alert
    }
    return output



# ---------- Output ----------
def output_fn(prediction, accept="application/json"):
    if accept != "application/json":
        raise ValueError(f"Unsupported response content type: {accept}")
    return json.dumps(prediction)





if __name__ == "__main__":
    import numpy as np
    from src.utils.kvs_stream import get_kvs_hls_url

    print("[INFO] Running in local video test mode...")

    # ---------- Setup ----------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FRAME_WARMUP_RUNS = 1
    model_dir = "."  # folder where best.pt is located
    model = model_fn(model_dir)
    print("[INFO] Model loaded successfully.")

    # ---------- Load video ----------
    video_path = r"C:\Users\uct\Desktop\AiCCTV\test_videos\istockphoto-1404365178-640_adpp_is.mp4"  # change this to your video filename
    url=get_kvs_hls_url("Cam424")
    # if not os.path.exists(video_path):
    #     raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video file")

    # Prepare video writer for output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    save_output = False # Set to False if you just want to view live without saving

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    if save_output:
        out = cv2.VideoWriter("annotated_video.mp4", fourcc, fps, (frame_width, frame_height))

    print("[INFO] Starting video inference... (press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB and PIL Image for YOLO input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        # Run inference
        output = predict_fn(image_pil, model)

        # Decode annotated frame
        annotated_bytes = base64.b64decode(output["annotated_frame"])
        nparr = np.frombuffer(annotated_bytes, np.uint8)
        annotated_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Show live
        cv2.imshow("PPE Detection", annotated_frame)

        # Save output frame
        if save_output and out:
            out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("[INFO] Video inference completed. Saved as annotated_video.mp4")

