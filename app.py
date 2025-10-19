from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tempfile, os

# =========================================
# Flask App Initialization
# =========================================
app = Flask(__name__)
CORS(app)

# =========================================
# Helper Functions
# =========================================
def compute_spectral_slope(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(h, w)
    crop = gray[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ps = np.abs(fshift)**2
    center = s // 2
    radial = []
    for r in range(1, center):
        y, x = np.ogrid[:s, :s]
        mask = ((y-center)**2 + (x-center)**2 >= (r-0.5)**2) & ((y-center)**2 + (x-center)**2 < (r+0.5)**2)
        vals = ps[mask]
        if vals.size:
            radial.append(np.mean(vals))
    radial = np.array(radial)
    if radial.size < 3:
        return 0.0
    freqs = np.arange(1, 1 + radial.size)
    logf = np.log(freqs)
    logp = np.log(radial + 1e-12)
    slope, _ = np.polyfit(logf, logp, 1)
    return float(slope)

def compute_noise_residual(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    residual = gray - blurred
    return float(np.var(residual))

def build_tf_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def detect_fake_probability(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128)).astype('float32') / 255.0
    inp = np.expand_dims(np.expand_dims(resized, axis=0), axis=-1)
    prob = float(model.predict(inp, verbose=0)[0][0])
    return prob

# =========================================
# Constants / Baselines
# =========================================
BASELINE_SLOPE = -3.0
AI_SLOPE_THRESHOLD = -3.4
tf_model = build_tf_model()

# =========================================
# Routes
# =========================================
@app.route("/")
def home():
    return {"status": "ok", "message": "AEGIS CV Server is running"}

@app.route("/analyze", methods=["POST"])
def analyze_video():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video file provided"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        file.save(tmp.name)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    sample_every_n = 10  # reduce CPU/memory load

    slope_sum = 0.0
    residual_sum = 0.0
    prob_sum = 0.0
    count = 0
    anomaly_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % sample_every_n != 0:
            continue

        slope = compute_spectral_slope(frame)
        residual = compute_noise_residual(frame)
        prob = detect_fake_probability(frame, tf_model)

        slope_sum += slope
        residual_sum += residual
        prob_sum += prob
        count += 1

        if slope < AI_SLOPE_THRESHOLD or residual < 1000:
            anomaly_frames.append(frame_count)

    cap.release()
    os.remove(video_path)

    avg_slope = slope_sum / count if count > 0 else BASELINE_SLOPE
    slope_dev = abs(avg_slope - BASELINE_SLOPE)
    avg_residual = residual_sum / count if count > 0 else 8000
    avg_prob = prob_sum / count if count > 0 else 0.5
    avg_fft_dev = (sum([abs(slope - BASELINE_SLOPE) * 1000 for slope in [avg_slope]])) / 1  # simplified

    spectral_weight = 0.7
    noise_weight = 0.2
    cnn_weight = 0.1
    slope_term = min(slope_dev / 0.5, 1.0)
    noise_term = max(0.0, min((10000 - avg_residual) / 10000, 1.0))
    cnn_term = avg_prob
    perceptual_score = (1 - (spectral_weight * slope_term + noise_weight * noise_term + cnn_weight * cnn_term))
    perceptual_score = round(max(0.0, min(perceptual_score, 1.0)), 4)

    explanation = (
        f"The video’s spectral slope is {avg_slope:.2f}, compared to the baseline mean of {BASELINE_SLOPE:.2f}. "
        f"Slope deviation = {slope_dev:.2f}. Noise variance = {avg_residual:.2f}. "
        f"AI artifact probability (TensorFlow): {avg_prob:.2f}."
    )

    result = {
        "perceptualScore": perceptual_score,
        "explanation": explanation,
        "forensicIndicators": {
            "avgFFTDeviation": avg_fft_dev,
            "avgNoiseResidual": avg_residual,
            "modelConfidence": avg_prob,
            "spectralSlope": avg_slope,
        },
        "anomalyFrames": anomaly_frames,
    }

    return jsonify(result)

# =========================================
# Run Server
# =========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
