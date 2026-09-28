# Setup-time image that turns the official BirdNET V2.4 Keras release into the ONNX files Birdsong
# runs (docs/MODEL.md). Used by scripts/fetch-models.sh; never part of the runtime image.
# Versions match the ones used to produce and verify the files in docs/MODEL.md.
FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir \
      tensorflow==2.21.0 \
      tf-keras==2.21.0 \
      tf2onnx==1.17.0 \
      onnx==1.22.0 \
      onnxruntime==1.30.0 \
      numpy==2.5.3 \
      soundfile==0.14.0 \
      h5py==3.14.0 \
      protobuf==7.36.1

ENV HOME=/tmp TF_CPP_MIN_LOG_LEVEL=2 TF_USE_LEGACY_KERAS=1
WORKDIR /work
COPY tools/convert_model/export_headless_v24.py /work/
COPY docker/convert-models.sh /work/

# Expects the unzipped BirdNET_v2.4_keras.zip in /models (read-write) and tools/fixtures in /fixtures.
ENTRYPOINT ["/bin/sh", "/work/convert-models.sh"]
