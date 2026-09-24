"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { healthCheck, predict } from "./lib/api";
import type { PredictResponse } from "./lib/types";

const FILE_INPUT_ID = "chest-xray-file";
const ACCEPTED_FILE_TYPES = ["image/jpeg", "image/png", "image/jpg"];
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

/**
 * The API client may gain AbortSignal support independently of this UI. Keep the
 * optional third argument type-safe while allowing the current two-argument
 * implementation to ignore it at runtime until that client lands.
 */
type PredictWithOptionalSignal = (
  file: File,
  onProgress?: (progress: number, stage: string) => void,
  signal?: AbortSignal
) => Promise<PredictResponse>;

/**
 * Simple progress bar component.
 */
function ProgressBar({
  value,
  label,
}: {
  value: number;
  label: string;
}) {
  const boundedValue = Math.min(100, Math.max(0, Math.round(value)));

  return (
    <div
      className="progress"
      role="progressbar"
      aria-label="Prediction progress"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={boundedValue}
      aria-valuetext={label || "Prediction in progress"}
      aria-live="polite"
      aria-atomic="true"
    >
      <div className="progress-track" aria-hidden="true">
        <div className="progress-fill" style={{ width: `${boundedValue}%` }} />
      </div>
      <div className="progress-label">{label}</div>
    </div>
  );
}

/**
 * Horizontal confidence bar for one label.
 */
function ConfidenceBar({
  label,
  value,
  color,
  highlight,
}: {
  label: string;
  value: number;
  color: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={`confidence-row ${highlight ? "confidence-row--highlight" : ""}`}
    >
      <span className="confidence-label">{label}</span>
      <div className="confidence-track">
        <div
          className="confidence-fill"
          style={{
            width: `${Math.round(value * 100)}%`,
            backgroundColor: color,
          }}
        />
      </div>
      <span className="confidence-value">{value.toFixed(2)}</span>
    </div>
  );
}

/**
 * Result card shown after a successful prediction.
 */
function ResultCard({ data }: { data: PredictResponse }) {
  const primaryColor =
    data.primary_prediction === "Pneumonia" ? "#dc2626" : "#16a34a";
  const isPneumonia = data.primary_prediction === "Pneumonia";

  return (
    <div className="result-card">
      <div className="result-primary">
        <div className="result-badge" style={{ backgroundColor: primaryColor }}>
          {data.primary_prediction}
        </div>
        <div className="result-confidence">
          <span className="result-confidence-label">Confidence</span>
          <span className="result-confidence-value">
            {(data.primary_confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {isPneumonia && (
        <div className="result-subtype">
          <span className="result-subtype-label">Subtype</span>
          <span
            className="result-subtype-value"
            style={{
              color:
                data.subtype_prediction === "Uncertain Pneumonia Subtype"
                  ? "#d97706"
                  : "#7c3aed",
            }}
          >
            {data.subtype_prediction}
          </span>
          {data.subtype_confidence !== null && (
            <span className="result-subtype-confidence">
              ({(data.subtype_confidence * 100).toFixed(1)}%)
            </span>
          )}
        </div>
      )}

      <div className="result-section">
        <span className="result-section-title">Probabilities</span>
        <div className="probabilities">
          {(["Normal", "Pneumonia"] as const).map((label) => (
            <ConfidenceBar
              key={label}
              label={label}
              value={data.probabilities[label] ?? 0}
              color={label === "Normal" ? "#16a34a" : "#dc2626"}
              highlight={label === data.primary_prediction}
            />
          ))}
          {isPneumonia && (
            <>
              <ConfidenceBar
                label="Bacterial Pneumonia"
                value={data.probabilities["Bacterial Pneumonia"] ?? 0}
                color="#7c3aed"
                highlight={
                  data.subtype_prediction === "Bacterial Pneumonia"
                }
              />
              <ConfidenceBar
                label="Viral Pneumonia"
                value={data.probabilities["Viral Pneumonia"] ?? 0}
                color="#0891b2"
                highlight={
                  data.subtype_prediction === "Viral Pneumonia"
                }
              />
            </>
          )}
        </div>
      </div>

      {/* Explainability heatmap */}
      {data.heatmap_b64 && <HeatmapOverlay heatmapSrc={data.heatmap_b64} />}

      <div className="disclaimer">
        <span className="disclaimer-icon">⚠️</span>
        <p>{data.disclaimer}</p>
      </div>
    </div>
  );
}

/**
 * Heatmap overlay component that displays a Grad-CAM heatmap
 * composited over the original chest X-ray image.
 */
function HeatmapOverlay({ heatmapSrc }: { heatmapSrc: string }) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapLoaded, setHeatmapLoaded] = useState(false);

  useEffect(() => {
    if (heatmapSrc) {
      const img = new Image();
      img.onload = () => setHeatmapLoaded(true);
      img.onerror = () => setHeatmapLoaded(false);
      img.src = heatmapSrc;
      return () => {
        img.onload = null;
        img.onerror = null;
      };
    }
  }, [heatmapSrc]);

  if (!heatmapSrc) {
    return null;
  }

  return (
    <div className="heatmap-section">
      <div className="heatmap-header">
        <span className="heatmap-title">Explainability Heatmap</span>
        <label className="heatmap-toggle">
          <input
            type="checkbox"
            checked={showHeatmap}
            onChange={(e) => setShowHeatmap(e.target.checked)}
          />
          <span>Show heatmap</span>
        </label>
      </div>
      <div className="heatmap-visual">
        {heatmapLoaded ? (
          <img
            src={heatmapSrc}
            alt="Grad-CAM heatmap overlaid on chest X-ray"
            className={showHeatmap ? "heatmap-image" : "heatmap-image--hidden"}
          />
        ) : (
          <div className="heatmap-loading">Loading heatmap...</div>
        )}
        {!showHeatmap && (
          <div className="heatmap-placeholder">
            Heatmap hidden — check the box above to reveal
          </div>
        )}
      </div>
      <p className="heatmap-caption">
        Grad-CAM visualization highlights the regions of the X-ray that most
        influenced the model&apos;s prediction. Red areas indicate higher importance.
      </p>
    </div>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");
  const [statusMessage, setStatusMessage] = useState(
    "Choose a chest X-ray image to begin."
  );
  const [error, setError] = useState<string | null>(null);
  const [pipelineReady, setPipelineReady] = useState<boolean | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const activeRequestIdRef = useRef(0);
  const requestControllerRef = useRef<AbortController | null>(null);
  const loadingRef = useRef(false);
  const fileRef = useRef<File | null>(null);

  /**
   * Revoke the current blob URL when the preview is replaced or the component
   * unmounts, so we never leak object URLs.
   */
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = null;
      }
    };
  }, []);

  /**
   * Check backend health on mount.
   */
  useEffect(() => {
    healthCheck()
      .then((h) => setPipelineReady(h.pipeline_ready))
      .catch(() => setPipelineReady(false));
  }, []);

  /**
   * Cancel any in-flight prediction when the page is left.
   */
  useEffect(() => {
    return () => {
      activeRequestIdRef.current++;
      requestControllerRef.current?.abort();
    };
  }, []);

  /**
   * Reset the native file input so selecting the same file again fires a new
   * change event after an invalid selection or removal.
   */
  function resetFileInput() {
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  /**
   * Invalidate callbacks from a superseded request and cancel its network work
   * when the API client supports AbortSignal.
   */
  function invalidateActiveRequest() {
    activeRequestIdRef.current += 1;
    requestControllerRef.current?.abort();
    requestControllerRef.current = null;
  }

  function cancelActiveRequest() {
    invalidateActiveRequest();
    loadingRef.current = false;
    setLoading(false);
  }

  function clearSelectedFile() {
    cancelActiveRequest();
    fileRef.current = null;
    setFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
    setProgressLabel("");
    setStatusMessage("No image selected.");

    const current = previewUrlRef.current;
    previewUrlRef.current = null;
    setPreview(null);
    if (current) {
      // Defer revocation so the img element has unloaded the src.
      requestAnimationFrame(() => URL.revokeObjectURL(current));
    }
    resetFileInput();
  }

  function validateSelectedFile(selected: File): string | null {
    if (!ACCEPTED_FILE_TYPES.includes(selected.type)) {
      return `Unsupported file type: ${selected.type || "unknown"}. Please upload a JPG or PNG image.`;
    }
    if (selected.size > MAX_FILE_SIZE_BYTES) {
      return `File too large: ${(selected.size / 1024 / 1024).toFixed(2)} MB. Maximum is 10 MB.`;
    }
    return null;
  }

  /**
   * Apply a file from either the native picker or a real drop event.
   */
  function applySelectedFile(selected: File) {
    if (loadingRef.current) {
      resetFileInput();
      return;
    }

    const validationError = validateSelectedFile(selected);
    if (validationError) {
      clearSelectedFile();
      setError(validationError);
      setStatusMessage(validationError);
      return;
    }

    cancelActiveRequest();
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }

    const nextPreview = URL.createObjectURL(selected);
    previewUrlRef.current = nextPreview;
    fileRef.current = selected;
    setFile(selected);
    setError(null);
    setResult(null);
    setProgress(0);
    setProgressLabel("");
    setPreview(nextPreview);
    setStatusMessage(`Selected ${selected.name}. Review the preview, then analyze the image.`);
  }

  /**
   * Handle file selection from the native file input.
   */
  function onFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    if (!selected) {
      return;
    }
    applySelectedFile(selected);
  }

  function onDragEnter(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (
      !loadingRef.current &&
      Array.from(event.dataTransfer.types).includes("Files")
    ) {
      setIsDragging(true);
      setStatusMessage("Release the image to upload it.");
    }
  }

  function onDragOver(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (!loadingRef.current) {
      event.dataTransfer.dropEffect = "copy";
    }
  }

  function onDragLeave(event: React.DragEvent<HTMLDivElement>) {
    if (
      event.relatedTarget instanceof Node &&
      event.currentTarget.contains(event.relatedTarget)
    ) {
      return;
    }
    setIsDragging(false);
    if (!loadingRef.current) {
      setStatusMessage(
        fileRef.current
          ? "Image selection unchanged."
          : "Choose a chest X-ray image to begin."
      );
    }
  }

  function onDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    if (loadingRef.current) {
      return;
    }
    const selected = event.dataTransfer.files?.[0];
    if (!selected) {
      setStatusMessage("No image was dropped.");
      return;
    }
    applySelectedFile(selected);
  }

  /**
   * Submit the selected file to the backend.
   */
  async function onPredict() {
    const selectedFile = fileRef.current;
    if (!selectedFile || loadingRef.current) {
      return;
    }

    cancelActiveRequest();
    const requestId = activeRequestIdRef.current;
    const requestController = new AbortController();
    requestControllerRef.current = requestController;
    loadingRef.current = true;
    setLoading(true);
    setError(null);
    setProgress(0);
    setProgressLabel("Starting...");
    setStatusMessage("Starting prediction.");

    const onProgress = (value: number, stage: string) => {
      if (requestId !== activeRequestIdRef.current) {
        return;
      }
      setProgress(value);
      setProgressLabel(stage);
      setStatusMessage(stage);
    };

    try {
      // The current API client remains contract-compatible with two arguments;
      // the optional signal is forwarded when the cancellable client is merged.
      const cancellablePredict = predict as unknown as PredictWithOptionalSignal;
      const data = await cancellablePredict(
        selectedFile,
        onProgress,
        requestController.signal
      );
      if (
        requestId !== activeRequestIdRef.current ||
        requestController.signal.aborted
      ) {
        return;
      }
      setResult(data);
      setProgress(100);
      setProgressLabel("Complete");
      setStatusMessage("Prediction complete.");
    } catch (err) {
      if (
        requestId !== activeRequestIdRef.current ||
        requestController.signal.aborted
      ) {
        return;
      }
      const message = err instanceof Error ? err.message : "Prediction failed";
      setError(message);
      setResult(null);
      setStatusMessage(message);
    } finally {
      if (requestId === activeRequestIdRef.current) {
        requestControllerRef.current = null;
        loadingRef.current = false;
        setLoading(false);
      }
    }
  }

  /**
   * Remove the current file and preview.
   */
  function onRemove() {
    clearSelectedFile();
  }

  return (
    <main className="page">
      <header className="header">
        <h1 className="header-title">
          🫁 Pneumonia Detection
        </h1>
        <p className="header-subtitle">
          Upload a chest X-ray for hierarchical AI analysis.
        </p>
      </header>

      {/* Health / connectivity indicator */}
      <div
        className="connectivity"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {pipelineReady === null ? (
          <span className="connectivity-status connectivity-status--checking">
            Checking backend...
          </span>
        ) : pipelineReady ? (
          <span className="connectivity-status connectivity-status--ok">
            Backend ready
          </span>
        ) : (
          <span className="connectivity-status connectivity-status--error">
            Backend unavailable — start the API with
            <code>uvicorn backend.app:app</code>
          </span>
        )}
      </div>

      {/* Upload area */}
      <section className="upload-section" aria-labelledby="upload-heading">
        <h2 id="upload-heading" className="visually-hidden">
          Upload a chest X-ray
        </h2>
        <div
          className={`dropzone ${file ? "dropzone--filled" : ""} ${isDragging ? "dropzone--dragging" : ""}`}
          onDragEnter={onDragEnter}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          aria-disabled={loading}
        >
          <label className="dropzone-label" htmlFor={FILE_INPUT_ID}>
            {preview ? (
              <img
                className="preview-image"
                src={preview}
                alt="Uploaded X-ray preview"
              />
            ) : (
              <>
                <span
                  className="dropzone-icon"
                  aria-hidden="true"
                >
                  📤
                </span>
                <span className="dropzone-text">
                  {loading
                    ? "Analyzing selected image..."
                    : "Choose or drop a chest X-ray image here"}
                </span>
                <span className="dropzone-hint">
                  JPG, JPEG, or PNG — up to 10 MB
                </span>
              </>
            )}
          </label>
          <input
            ref={fileInputRef}
            id={FILE_INPUT_ID}
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            onChange={onFileChange}
            className="file-input"
            disabled={loading}
            aria-label="Chest X-ray image"
            aria-busy={loading}
            aria-describedby="upload-help upload-privacy"
          />
          <span id="upload-help" className="visually-hidden">
            Use the file picker or drag and drop. Supported formats are JPG, JPEG,
            and PNG, with a maximum size of 10 megabytes.
          </span>
          {preview && (
            <button
              className="remove-button"
              onClick={onRemove}
              type="button"
              disabled={loading}
              aria-label="Remove selected image"
            >
              Remove
            </button>
          )}
        </div>

        <div
          id="upload-status"
          className="visually-hidden"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          {statusMessage}
        </div>
        {error && (
          <div
            className="error-banner"
            role="alert"
            aria-live="assertive"
          >
            {error}
          </div>
        )}

        {file && !loading && !result && (
          <button
            className="predict-button"
            onClick={onPredict}
            disabled={pipelineReady === false}
            aria-describedby="upload-status"
          >
            Analyze Image
          </button>
        )}

        {loading && (
          <>
            <button
              className="predict-button predict-button--disabled"
              type="button"
              disabled
            >
              Analyzing...
            </button>
            <ProgressBar value={progress} label={progressLabel} />
          </>
        )}

        <aside
          id="upload-privacy"
          className="privacy-notice"
          aria-label="Privacy and data transfer"
        >
          <h2 className="privacy-title">Privacy and data transfer</h2>
          <p>
            When you choose <strong>Analyze Image</strong>, the image is sent from
            your browser to this application&apos;s configured backend. The backend
            is designed to process requests without persistently storing the image
            or prediction, but configured third-party model services may apply
            their own retention and processing terms. Do not upload
            patient-identifiable images or protected health information.
          </p>
        </aside>
      </section>

      {/* Result */}
      {result && <ResultCard data={result} />}

      {/* Footer */}
      <footer className="footer">
        <div className="footer-inner">
          <p>
            Powered by a hierarchical classification pipeline: Normal vs Pneumonia,
            then Bacterial vs Viral subtype.
          </p>
          <Link href="/research" className="footer-link">
            View research results
          </Link>
        </div>
      </footer>
    </main>
  );
}
