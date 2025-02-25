from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO Model
model = YOLO("best.pt")

def predict_image(image_path):
    results = model(image_path)
    detections = results[0].boxes.data  # Bounding box data
    classes = results[0].names  # Class names
    
    detected_results = []
    for det in detections:
        confidence = float(det[4])  # Confidence score
        class_id = int(det[5])  # Class ID from model output
        disease_name = classes[class_id]
        
        detected_results.append({"disease": disease_name, "confidence": confidence})

    return detected_results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No selected file")

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Run YOLO prediction
            predictions = predict_image(file_path)
            return render_template("index.html", image=file_path, predictions=predictions)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
