"""Run the reference BirdNET V2.4 *meta* (location/week) TFLite model and write golden outputs.

Input is [latitude, longitude, week] as float32; week is 1..48 (4 per month) or -1 for year-round.
Output is one occurrence probability per class (6522). Dev-time tool only.
"""
import argparse, json, sys
import numpy as np
import tensorflow as tf

CASES = [
    {"name": "boston_week20", "lat": 42.36, "lon": -71.06, "week": 20},
    {"name": "boston_yearround", "lat": 42.36, "lon": -71.06, "week": -1},
    {"name": "berlin_week1", "lat": 52.52, "lon": 13.40, "week": 1},
    {"name": "sydney_week30", "lat": -33.87, "lon": 151.21, "week": 30},
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../../models/meta-model.tflite")
    ap.add_argument("--out", default="../fixtures/golden/meta_v24.json")
    a = ap.parse_args()
    it = tf.lite.Interpreter(model_path=a.model)
    it.allocate_tensors()
    inp = it.get_input_details()[0]; out = it.get_output_details()[0]
    cases = []
    for c in CASES:
        x = np.array([[c["lat"], c["lon"], c["week"]]], dtype=np.float32)
        it.set_tensor(inp["index"], x); it.invoke()
        p = it.get_tensor(out["index"])[0].astype(np.float32)
        allowed = int((p >= 0.03).sum())
        print(f'{c["name"]:18s} allowed@0.03={allowed:5d} max={p.max():.3f} argmax={int(p.argmax())}', file=sys.stderr)
        cases.append({**c, "probabilities": [float(v) for v in p], "allowed_at_0_03": allowed})
    json.dump({"model": a.model.split("/")[-1], "threshold_default": 0.03, "cases": cases}, open(a.out, "w"))
    print("wrote", a.out, file=sys.stderr)

if __name__ == "__main__":
    main()
