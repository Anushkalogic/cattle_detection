from flask import Flask, render_template, send_file, request, jsonify, url_for
from inference import run_roboflow_pipeline
from database import init_db, fetch_all_detections, query_by_label
import os
import csv

app = Flask(__name__)
init_db()



@app.route('/')
def index():
    print("🔁 Starting pipeline")
    run_roboflow_pipeline()
    print("✅ Pipeline done")
    return render_template("video_result.html", video_path="output/temp_output.mp4")

@app.route('/download-csv')
def download_csv():
    rows = fetch_all_detections()
    csv_path = "static/detection_report.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Path', 'Cow', 'Stranger Cow', 'Dog', 'Timestamp'])
        for row in rows:
            writer.writerow(row[1:])  # skip ID

    return send_file(csv_path, as_attachment=True)

@app.route('/search')
def search():
    label = request.args.get("q")
    if not label:
        return jsonify({"error": "Missing ?q=..."}), 400
    results = query_by_label(label)
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True)
