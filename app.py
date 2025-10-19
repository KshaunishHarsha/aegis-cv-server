from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import tempfile, os

app = Flask(__name__)

@app.route("/")
def home():
    return {"status": "ok", "message": "AEGIS CV Server is running"}

@app.route("/analyze", methods=["POST"])
def analyze_video():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video file provided"}), 400

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        file.save(tmp.name)
        video_path = tmp.name

    # --- Perceptual analysis placeholder (replace with your Colab logic) ---
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    spectral_slope = -3.07
    noise_variance = 8693.5
    ai_probability = float(tf.random.uniform([], 0, 1))
    cap.release()
    os.remove(video_path)

    # Build result
    result = {
        "perceptualScore": round((1 - ai_probability) * 0.8 + 0.2, 4),
        "explanation": (
            f"The video’s spectral slope is {spectral_slope:.2f}, noise variance {noise_variance:.2f}, "
            f"AI artifact probability (TensorFlow): {ai_probability:.2f}."
        ),
        "forensicIndicators": {
            "avgFFTDeviation": 74.26,
            "avgNoiseResidual": noise_variance,
            "modelConfidence": ai_probability,
            "spectralSlope": spectral_slope,
        },
        "anomalyFrames": [],
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
