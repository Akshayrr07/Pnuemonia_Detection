# Limitations & Future Scope

## Current Limitations

### 1. Not a medical device

This system is an educational and research tool. It does not provide a medical diagnosis and should never be used as the sole basis for clinical decisions. The included disclaimer in every prediction response makes this explicit, but it is worth stating directly: no regulatory clearance (FDA, CE, etc.) has been obtained, and the system has not been validated for clinical use.

### 2. Model performance is imperfect, especially for subtypes

The system is hierarchical by design because the data supports it that way:

- **Binary (Normal vs Pneumonia):** 93.78% accuracy. The strongest part of the system.
- **Subtype (Bacterial vs Viral):** 75.67% accuracy. Harder, and the response should be read as "the model's best guess" rather than a definitive answer.
- **Direct three-class:** 74–77% accuracy. The viral class is the hardest to separate, and forcing a single model to do everything at once is what motivated the hierarchical design.

A user looking at a "Bacterial Pneumonia" prediction with 0.55 confidence should not treat that the same as one at 0.90 confidence. The probabilities are exposed so a human can calibrate.

### 3. The quality of the hosted model matters

The production path calls the Hugging Face Inference API. Whatever model is hosted at the configured repo ids is what the system uses. If the hosted checkpoints are not the same ones that produced the research metrics, the live accuracy will differ from what is advertised on the research page.

The research metrics in this repository (93.78% binary, 75.67% subtype) were produced during offline experimentation. They describe the research pipeline, not necessarily the identically-configured live endpoint.

### 4. Dataset scope and provenance are limited

The checked-in registry contains 5,891 records from two current source labels (`dataset_1` and `dataset_2`); a historical duplicate log also contains `dataset_3` records that are not in the current registry. Exact source URLs, licenses, retrieval versions, and a source manifest are unresolved, so the earlier three-source description is not verified. The research split is 70/15/15, but it is not a large, multi-site, demographically diverse clinical corpus. Sensitivity to age, sex, device manufacturer, view position (AP vs PA), and co-morbidities is not separately characterized.

### 5. Image input assumptions

The preprocessing expects a chest X-ray in JPEG or PNG form. Extremely unusual image orientations, non-standard resolutions, or images that are not true chest X-rays will still be processed but the result may not be meaningful. The backend validates file type, extension, size, and decodability — but it cannot validate that the image is, in fact, a chest X-ray of a human.

### 6. No persistent storage or audit log

The backend is stateless between requests. It does not store uploaded images, predictions, or user information. This is a deliberate privacy choice, but it also means:

- There is no audit trail for predictions.
- There is no way to review a previous result.
- There is no feedback loop from clinical outcomes to model improvement.

This stateless design is not a privacy guarantee for a deployed environment. Browser, reverse proxy, container, hosting, and model-provider logs may retain requests or image bytes. Do not use the public demo with real patient data without an approved privacy, security, consent, retention, and access-control review.

### 7. Explainability is gated behind a local checkpoint

Grad-CAM is implemented (`src/inference/explainability.py`) and wired into the backend, but it only runs when `LOCAL_MODEL_PATH` points to a real checkpoint file. Without a checkpoint, `heatmap_b64` is always `null` and the heatmap overlay in the frontend has nothing to show.

Hugging Face Inference API is a black-box HTTP endpoint — it returns labels and scores but not internal activations or gradients, so Grad-CAM cannot run on the hosted path.

### 8. Deployment environment specifics

- **Frontend** is deployed to Cloudflare Pages as a static export. This is fast and cheap, but it means the frontend cannot do server-side rendering, serverless functions, or direct backend calls without configuring `NEXT_PUBLIC_BACKEND_URL` **in the frontend build environment** and rebuilding the static export. When that build-time var is unset, the frontend falls back to same-origin, which is only correct if the backend is served from the same domain.
- **Backend** is containerized with Docker. The Dockerfile is known to work on this machine. It has not been tested on a different host OS or a different cloud container runtime. The healthcheck, env var injection, and CORS config should all be re-verified when deploying to a new environment.
- **Docker build size (~320 MB)** is acceptable for a Python ML service but could be reduced with a multi-arch or distroless runtime if image size becomes a deployment constraint.

## Future Scope

### Near-term (next iteration)

1. **Upload trained checkpoints to Hugging Face** — create model repos, upload the checkpoints that correspond to the research metrics, and set `HF_BINARY_MODEL_ID` / `HF_SUBTYPE_MODEL_ID` so the live endpoint reflects the reported numbers.

2. **Threshold tuning** — the current 0.50 thresholds were chosen as reasonable defaults. A proper threshold search on the validation set (maximizing sensitivity for the binary task, or balancing precision/recall for the subtype task) would produce more defensible operating points and should be documented per-model.

3. **Calibration** — the confidence numbers returned by the hosted model may not be calibrated probabilities. A calibration layer (temperature scaling or Platt scaling on validation outputs) would make the confidence values more trustworthy.

4. **ROC/AUC, sensitivity, specificity** — the research page reports accuracy. For a medical-adjacent tool, sensitivity (true positive rate for pneumonia) and specificity (true negative rate) are more relevant and should be computed from the held-out test set and surfaced somewhere in the documentation.

5. **Model cards** — each hosted checkpoint should have a model card describing training data, metrics, known failure modes, and intended use. This is currently absent.

### Medium-term

6. **Better subtype model** — the 75.67% subtype accuracy is the clearest room-for-improvement signal. Options: more/better data for the bacterial-vs-viral distinction, pretrained chest X-ray models fine-tuned on this task, or an external validation set to confirm the result is not overfit to this particular Kaggle corpus.

7. **Ensemble in production** — the repository has soft-voting and weighted-voting ensemble utilities in `experiments/`. With multiple checkpoints available, an ensemble endpoint could improve robustness. The current production contract (`HierarchicalPrediction`) already supports this without changing the API shape.

8. **Image-level explainability in the hosted path** — since HF Inference API does not expose gradients, true Grad-CAM for hosted models is not possible without a different serving approach. Alternatives: self-hosted inference with a framework that does expose intermediate activations, or saliency-map approximations at the API level.

9. **Audit log (privacy-preserving)** — a log of prediction timestamps, model versions, and coarse result categories (not the images themselves) would support monitoring and drift detection without storing patient data.

10. **Input sanity checks** — at minimum: verify the uploaded image is plausibly a chest X-ray (aspect ratio, file size distribution, basic content heuristics) before running inference, and return a clearer warning when it is not.

### Long-term

11. **Clinical validation** — any move toward real clinical use would require a proper validation study on an independent dataset, with Institutional Review Board (IRB) approval where applicable, and regulatory review.

12. **Continuous monitoring** — track prediction distributions, latency, error rates, and model version over time. Alert on distribution shift or sudden degradation.

13. **Multi-view and demographic handling** — extend the dataset and evaluation to characterize and, if possible, improve performance across different view positions and demographic groups.
