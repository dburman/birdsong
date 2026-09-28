"""Export BirdNET V2.4 *without* its in-graph mel-spectrogram layers (path (c) of the spike).

The Keras model is: INPUT(144000) -> MEL_SPEC1, MEL_SPEC2 -> concatenate(96,511,2) -> CNN -> 6522 logits.
tract cannot run the STFT ops, so we (1) export the CNN from `concatenate` onward to ONNX and
(2) dump everything the Rust mel frontend needs to reproduce MEL_SPEC1/2 exactly:
mel filterbanks, magnitude_scaling weights, hyperparameters, and reference spectrograms.

Dev-time tool only. Outputs go to --out-dir (default ../../models).
"""
import argparse, json, os, sys
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tf2onnx, onnx
import soundfile as sf

def load_mel_layer_class(models_dir):
    src = open(os.path.join(models_dir, "MelSpecLayerSimple.py"), newline=None).read()
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    src = src.replace("import tensorflow as tf", "import tensorflow as tf\nimport tf_keras\ntf.keras = tf_keras")
    ns = {}
    exec(compile(src, "MelSpecLayerSimple.py", "exec"), ns)
    return ns["MelSpecLayerSimple"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="../../models")
    ap.add_argument("--out-dir", default="../../models")
    ap.add_argument("--wav", default="../fixtures/soundscape.wav")
    ap.add_argument("--golden", default="../fixtures/golden/soundscape.json")
    ap.add_argument("--no-reference", action="store_true",
                    help="do not write the reference spectrogram next to the golden file")
    a = ap.parse_args()

    Mel = load_mel_layer_class(a.models_dir)
    m = keras.models.load_model(os.path.join(a.models_dir, "audio-model.h5"),
                                custom_objects={"MelSpecLayerSimple": Mel}, compile=False)
    print("final activation:", m.get_layer("CLASS_ACTIVATION").get_config().get("activation"), file=sys.stderr)

    # --- 1. frontend model: waveform -> concatenated spectrogram (TF reference) ---
    concat = m.get_layer("concatenate").output
    frontend = keras.Model(inputs=m.input, outputs=concat, name="frontend")

    # --- 2. headless model: spectrogram -> logits ---
    spec_in = keras.Input(shape=concat.shape[1:], name="SPEC_INPUT")
    x = spec_in
    # Re-run every layer after `concatenate` in topological order. The tail is a plain chain
    # except for residual/pool blocks, so we rebuild by walking the original graph.
    layer_outputs = {m.get_layer("concatenate").name: x}
    for layer in m.layers[4:]:
        inbound = layer._inbound_nodes[0]
        srcs = inbound.inbound_layers
        if not isinstance(srcs, (list, tuple)):
            srcs = [srcs]
        inputs = [layer_outputs[s.name] for s in srcs]
        y = layer(inputs[0] if len(inputs) == 1 else inputs)
        layer_outputs[layer.name] = y
    headless = keras.Model(inputs=spec_in, outputs=layer_outputs["CLASS_DENSE_LAYER"], name="headless")  # logits, like the TFLite export

    # --- 3. verify frontend+headless == full model on a few chunks, against golden ---
    sig, sr = sf.read(a.wav, dtype="float32", always_2d=True); sig = sig.mean(axis=1)
    golden = json.load(open(a.golden))
    max_err = 0.0
    specs = []
    for c in golden["chunks"][:5]:
        s = c["start_sample"]; chunk = sig[s:s+144000]
        chunk = np.pad(chunk, (0, 144000-len(chunk)))[np.newaxis, :]
        spec = frontend.predict(chunk, verbose=0)
        specs.append(spec[0])
        logits = headless.predict(spec, verbose=0)[0]
        err = float(np.max(np.abs(logits - np.array(c["logits"]))))
        max_err = max(max_err, err)
    print(f"keras frontend+headless vs tflite golden (5 chunks): max|dlogit| = {max_err:.4f}", file=sys.stderr)
    if max_err > 0.05:
        sys.exit(f"conversion check failed: logits differ from the TFLite reference by {max_err:.4f}")

    # --- 4. export ONNX ---
    onnx_path = os.path.join(a.out_dir, "birdnet-v2.4-headless.onnx")
    spec_fn = tf.function(lambda x: headless(x), input_signature=[tf.TensorSpec([1, 96, 511, 2], tf.float32, name="spec")])
    model_proto, _ = tf2onnx.convert.from_function(spec_fn, input_signature=[tf.TensorSpec([1, 96, 511, 2], tf.float32, name="spec")], opset=13, output_path=onnx_path)
    print("onnx ops:", sorted({n.op_type for n in model_proto.graph.node}), file=sys.stderr)
    print("wrote", onnx_path, os.path.getsize(onnx_path)//1024, "KB", file=sys.stderr)

    # --- 5. dump frontend parameters ---
    params = {}
    for name in ["MEL_SPEC1", "MEL_SPEC2"]:
        L = m.get_layer(name)
        params[name] = {
            "sample_rate": L.sample_rate, "frame_length": L.frame_length, "frame_step": L.frame_step,
            "n_mels": L.spec_shape[0], "fmin": L.fmin, "fmax": L.fmax,
            "magnitude_scaling": float(L.mag_scale.numpy()),
            "mel_filterbank_shape": list(L.mel_filterbank.shape),
        }
        L.mel_filterbank.numpy().astype(np.float32).tofile(os.path.join(a.out_dir, f"{name}_melfb.f32"))
    json.dump(params, open(os.path.join(a.out_dir, "birdnet-v2.4-frontend.json"), "w"), indent=2)
    print(json.dumps(params, indent=2), file=sys.stderr)

    # --- 6. reference spectrogram of chunk 0 for the Rust frontend test (binary f32, NHWC (96,511,2)) ---
    if a.no_reference:
        return
    stem = os.path.splitext(os.path.basename(a.wav))[0]
    spec_path = os.path.join(os.path.dirname(a.golden), f"{stem}_chunk0_spec.f32")
    specs[0].astype(np.float32).tofile(spec_path)
    print("wrote", spec_path, specs[0].shape, file=sys.stderr)

if __name__ == "__main__":
    main()
