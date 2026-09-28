"""Run the reference BirdNET V2.4 TFLite model over fixture WAVs and write golden logits.

Dev-time tool only. Usage:
    python export_reference.py --model ../../models/audio-model.tflite \
        --wav ../fixtures/soundscape.wav --out ../fixtures/golden/soundscape.json [--max-chunks N]
"""
import argparse, json, sys, time
import numpy as np
import soundfile as sf
import tensorflow as tf

SAMPLE_RATE = 48000
CHUNK = 3 * SAMPLE_RATE

def load_wav(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz, got {sr}"
    return data.mean(axis=1)

def chunks(sig, overlap=0.0):
    step = int((3.0 - overlap) * SAMPLE_RATE)
    for start in range(0, len(sig), step):
        c = sig[start:start + CHUNK]
        if len(c) < 1.5 * SAMPLE_RATE:
            break
        if len(c) < CHUNK:
            c = np.pad(c, (0, CHUNK - len(c)))
        yield start, c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    it = tf.lite.Interpreter(model_path=a.model, num_threads=a.threads)
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    it.resize_tensor_input(inp["index"], [1, CHUNK])
    it.allocate_tensors()

    sig = load_wav(a.wav)
    results = []
    times = []
    for i, (start, c) in enumerate(chunks(sig)):
        if a.max_chunks and i >= a.max_chunks:
            break
        it.set_tensor(inp["index"], c[np.newaxis, :].astype(np.float32))
        t0 = time.perf_counter()
        it.invoke()
        times.append(time.perf_counter() - t0)
        logits = it.get_tensor(out["index"])[0].astype(np.float32)
        top5 = np.argsort(-logits)[:5]
        results.append({
            "chunk_index": i,
            "start_sample": start,
            "top5": [{"class_index": int(k), "logit": float(logits[k])} for k in top5],
            "logits": [float(x) for x in logits],
        })
        print(f"chunk {i:3d} start={start:8d} top1={top5[0]} logit={logits[top5[0]]:.3f}", file=sys.stderr)

    golden = {
        "model": a.model.split("/")[-1],
        "wav": a.wav.split("/")[-1],
        "sample_rate": SAMPLE_RATE,
        "chunk_samples": CHUNK,
        "num_classes": int(len(results[0]["logits"])) if results else 0,
        "chunks": results,
    }
    with open(a.out, "w") as f:
        json.dump(golden, f)
    print(f"wrote {a.out}: {len(results)} chunks, mean invoke {1000*np.mean(times):.1f} ms", file=sys.stderr)

if __name__ == "__main__":
    main()
