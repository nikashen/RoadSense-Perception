# Model Artifact Protocol

No model is loaded implicitly by importing `roadsense`. A future detector or
segmenter adapter must receive an explicit local artifact manifest and verify
it before inference.

An artifact manifest should bind:

- task and class ontology;
- framework/backend and version;
- checkpoint or ONNX file SHA-256;
- preprocessing, resize/letterbox, normalization, and coordinate convention;
- ONNX opset or exported graph metadata when applicable;
- dependency lock and runtime device;
- calibration/quantization settings and expected output schema.

The verifier must fail closed on a missing file, hash mismatch, unknown class,
unsupported opset, or incompatible output shape. Browser display controls may
change overlays but can never mutate the frozen artifact or evaluation report.

Weights and private training runs are local inputs. They are excluded from Git,
source distributions, Pages, and releases unless their license explicitly
allows redistribution and a separate publication review approves them.
