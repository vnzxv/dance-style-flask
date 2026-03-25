import os, json
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Always look in the same folder as app.py
BASE = os.path.dirname(os.path.abspath(__file__))

bundle  = joblib.load(os.path.join(BASE, "dance_classifier.pkl")) \
          if os.path.exists(os.path.join(BASE, "dance_classifier.pkl")) else None
results = json.load(open(os.path.join(BASE, "model_results.json"))) \
          if os.path.exists(os.path.join(BASE, "model_results.json")) else {}

FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

DANCE_INFO = {
    "Hip Hop":      {"emoji": "🎤", "color": "#e05c5c", "desc": "High energy, heavy bass, rhythmic footwork and freestyle flow."},
    "Contemporary": {"emoji": "💃", "color": "#2aada4", "desc": "Fluid, expressive movement blending ballet and jazz techniques."},
    "Ballet":       {"emoji": "🩰", "color": "#9b7fe8", "desc": "Classical precision, elegant lines and graceful choreography."},
    "K-Pop":        {"emoji": "⭐", "color": "#d4a017", "desc": "High-energy group choreography with synchronized, sharp movements."},
    "Salsa":        {"emoji": "🌶️",  "color": "#d4680a", "desc": "Passionate partner dance with Latin rhythm and hip motion."},
    "Breaking":     {"emoji": "🔥", "color": "#05a87a", "desc": "Athletic street dance with power moves, freezes and footwork."},
}

FEATURE_RANGES = {
    "danceability":     {"min": 0.0,   "max": 1.0,   "step": 0.01, "default": 0.65,  "label": "Danceability",     "unit": "0–1"},
    "energy":           {"min": 0.0,   "max": 1.0,   "step": 0.01, "default": 0.70,  "label": "Energy",           "unit": "0–1"},
    "loudness":         {"min": -60.0, "max": 5.0,   "step": 0.1,  "default": -7.0,  "label": "Loudness",         "unit": "dB"},
    "speechiness":      {"min": 0.0,   "max": 1.0,   "step": 0.01, "default": 0.08,  "label": "Speechiness",      "unit": "0–1"},
    "acousticness":     {"min": 0.0,   "max": 1.0,   "step": 0.01, "default": 0.15,  "label": "Acousticness",     "unit": "0–1"},
    "instrumentalness": {"min": 0.0,   "max": 1.0,   "step": 0.01, "default": 0.05,  "label": "Instrumentalness", "unit": "0–1"},
    "liveness":         {"min": 0.0,   "max": 1.0,   "step": 0.01, "default": 0.15,  "label": "Liveness",         "unit": "0–1"},
    "valence":          {"min": 0.0,   "max": 1.0,   "step": 0.01, "default": 0.60,  "label": "Valence",          "unit": "0–1"},
    "tempo":            {"min": 40.0,  "max": 220.0, "step": 0.5,  "default": 120.0, "label": "Tempo (BPM)",      "unit": "BPM"},
}


@app.route("/")
def index():
    model_name = bundle["model_name"]              if bundle else "Model not loaded"
    model_acc  = f"{bundle['test_accuracy']:.2%}"  if bundle else "—"
    return render_template("index.html",
                           model_name=model_name,
                           model_acc=model_acc,
                           dance_info=DANCE_INFO)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction  = None
    confidence  = None
    all_probs   = None
    form_values = {k: v["default"] for k, v in FEATURE_RANGES.items()}

    if request.method == "POST":
        try:
            features = []
            for feat in FEATURE_COLS:
                val = float(request.form.get(feat, FEATURE_RANGES[feat]["default"]))
                features.append(val)
                form_values[feat] = val

            X = np.array(features).reshape(1, -1)

            if bundle:
                pipe       = bundle["pipeline"]
                le         = bundle["label_encoder"]
                pred_idx   = pipe.predict(X)[0]
                prediction = le.inverse_transform([pred_idx])[0]

                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    probs      = pipe.predict_proba(X)[0]
                    confidence = float(probs.max())
                    all_probs  = {le.inverse_transform([i])[0]: float(p)
                                  for i, p in enumerate(probs)}
                    all_probs  = dict(sorted(all_probs.items(), key=lambda x: -x[1]))
            else:
                prediction = "⚠️ Place dance_classifier.pkl in the same folder as app.py"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("predict.html",
                           feature_ranges=FEATURE_RANGES,
                           form_values=form_values,
                           prediction=prediction,
                           confidence=confidence,
                           all_probs=all_probs,
                           dance_info=DANCE_INFO)


@app.route("/results")
def results_page():
    best = results.get("_best", "—")
    rows = {k: v for k, v in results.items() if not k.startswith("_")}
    rows = dict(sorted(rows.items(), key=lambda x: x[1]["test_acc"], reverse=True))
    return render_template("results.html", rows=rows, best=best)


if __name__ == "__main__":
    app.run(debug=True)
