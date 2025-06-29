from flask import Flask, render_template
import os, cv2, threading, subprocess
from database import init_db, insert_detection  # database methods

app = Flask(__name__)

# Paths
INPUT_VIDEO = r"static/dataset/video.mp4"
OUTPUT_VIDEO = r"static/output/output_video.mp4"
TEMP_OUTPUT = r"static/output/temp_output.mp4"
IMAGE_SAVE_DIR = r"static/detected_images"
lock = threading.Lock()
frame_count = 0
out = None  # Will be initialized later

# Ensure necessary folders
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

def run_roboflow_pipeline():
    global frame_count, out
    frame_count = 0

    # Clean old images
    for f in os.listdir(IMAGE_SAVE_DIR):
        os.remove(os.path.join(IMAGE_SAVE_DIR, f))

    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open input video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Initialize VideoWriter
    out = cv2.VideoWriter(TEMP_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    print("📽️ VideoWriter initialized")

    # Start Roboflow Inference
    from inference import InferencePipeline
    pipeline = InferencePipeline.init_with_workflow(
        api_key="IYnVxkCFFkQgBsrmcygz",
        workspace_name="cattle-wtx39",
        workflow_id="detect-count-and-visualize-6",
        video_reference=INPUT_VIDEO,
        max_fps=30,
        on_prediction=my_sink
    )

    print("🚀 Starting pipeline...")
    pipeline.start()
    pipeline.join()
    print("✅ Pipeline finished.")
    out.release()
    # Convert video with ffmpeg if valid
    if os.path.exists(TEMP_OUTPUT):
        size = os.path.getsize(TEMP_OUTPUT)
        print(f"📦 TEMP_OUTPUT size: {size} bytes")
        if size > 1000:
            subprocess.call([
                "ffmpeg", "-y", "-i", TEMP_OUTPUT,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                OUTPUT_VIDEO
            ])
            print("🎬 Final OUTPUT_VIDEO generated.")
        else:
            print(f"⚠️ TEMP_OUTPUT too small ({size} bytes) — skipping ffmpeg.")
    else:
        print("⚠️ TEMP_OUTPUT not found.")
def my_sink(result, video_frame):
    global frame_count, out

    output_image = result.get("output_image")

    if not output_image:
        print("⚠️ No output image in result.")
        return

    frame = output_image.numpy_image

    # Add even if there are NO predictions
    cow = stranger = dog = 0
    predictions = result.get("predictions", [])

    for pred in predictions:
        if isinstance(pred, tuple):
            pred = pred[0]
        if isinstance(pred, dict):
            cls = pred.get("class", "").lower()
            x1 = int(pred["x"] - pred["width"] / 2)
            y1 = int(pred["y"] - pred["height"] / 2)
            x2 = int(pred["x"] + pred["width"] / 2)
            y2 = int(pred["y"] + pred["height"] / 2)

            color = (0, 255, 0) if cls == 'cow' else (0, 0, 255) if cls == 'stranger_cow' else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, cls, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if cls == 'cow':
                cow += 1
            elif cls == 'stranger_cow':
                stranger += 1
            elif cls == 'dog':
                dog += 1

    overlay = f"Cows: {cow} | Strangers: {stranger} | Dogs: {dog}"
    cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    with lock:
        frame_count += 1
        image_path = os.path.join(IMAGE_SAVE_DIR, f"frame_{frame_count}.jpg")
        cv2.imwrite(image_path, frame)

        # ✅ FORCE writing every frame to video output, even if no detection
        if out:
            out.write(frame)
        else:
            print("❌ VideoWriter not initialized")

        # Save frame data to DB even if all counts are zero
        insert_detection(cow_count=cow, image_path=image_path)

        print(f"✅ Frame {frame_count} written: cow={cow}, stranger={stranger}, dog={dog}")

@app.route('/')
def index():
    run_roboflow_pipeline()
    return render_template("video_result.html", video_path="output/output_video.mp4")

@app.route('/get_dashboard_data')
def get_dashboard_data():
    return {"message": "ok"}, 200

if __name__ == '__main__':
    init_db()
    print("✅ DB Initialized")
    app.run(debug=True)
