import os
import cv2
import requests
import base64
from database import insert_detection

INPUT_VIDEO = "static/dataset/video.mp4"
DETECTED_DIR = "static/detected_frames"
os.makedirs(DETECTED_DIR, exist_ok=True)

def call_roboflow_api(image_path):
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
            img_base64 = base64.b64encode(img_bytes).decode()

        payload = {
            "api_key": "DiBsOHUZVRTHIOZjUoWJ",  # Replace with your key
            "inputs": {
                "image": {"type": "base64", "value": img_base64}
            }
        }

        response = requests.post(
            "https://detect.roboflow.com/anushka-t2wnn/detect-count-and-visualize-6",
            json=payload
        )

        if response.status_code == 200:
            return response.json()
        else:
            print("❌ Roboflow API Error:", response.status_code, response.text)
            return None
    except Exception as e:
        print(f"❌ Error during API call: {e}")
        return None

def run_roboflow_pipeline(frame_skip=10):
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {INPUT_VIDEO}")
        return

    success, frame = cap.read()
    if not success:
        print("❌ Could not read initial frame.")
        return

    height, width = frame.shape[:2]
    output_path = "static/output/temp_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_id = 0

    print("🚀 Starting Cattle Detection Pipeline...")

    while True:
        success, frame = cap.read()
        if not success:
            print("📉 No more frames or error reading frame.")
            break

        cow = stranger = dog = 0

        if frame_id % frame_skip == 0:
            frame_path = os.path.join(DETECTED_DIR, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)

            result = call_roboflow_api(frame_path)
            if result:
                predictions = result.get("results", {}).get("predictions", [])

                for pred in predictions:
                    cls = pred["class"].lower()
                    x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
                    x1, y1 = x - w // 2, y - h // 2
                    x2, y2 = x + w // 2, y + h // 2

                    color = (0, 255, 0) if cls == 'cow' else (0, 0, 255) if cls == 'stranger_cow' else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if cls == 'cow':
                        cow += 1
                    elif cls == 'stranger_cow':
                        stranger += 1
                    elif cls == 'dog':
                        dog += 1

                insert_detection(frame_path, cow, stranger, dog)

        # Overlay detection counts on top left
        overlay_text = f"Cows: {cow} | Stranger Cows: {stranger} | Dogs: {dog}"
        cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    print("✅ Detection pipeline completed.")
