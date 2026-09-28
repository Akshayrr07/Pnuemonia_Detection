"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { healthCheck, predict } from "./lib/api";
import type { PredictResponse } from "./lib/types";

function IndeterminateProgress({ label }: { label: string }) {
  return (
    <div className="progress" role="status" aria-live="polite">
      <progress aria-label={label} />
      <div className="progress-label">
        <strong>{label}</strong> — this stage is indeterminate because the API
        does not report upload or inference progress.
      </div>
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
          // Native <img> is intentional: the source is a data URL generated by the API.
          // eslint-disable-next-line @next/next/no-img-element
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
            Heatmap hidden &mdash; check the box above to reveal
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
  const [predictionStage, setPredictionStage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pipelineReady, setPipelineReady] = useState<boolean | null>(null);
  const [healthChecking, setHealthChecking] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const healthRequestRef = useRef(0);
  const predictionRequestRef = useRef(0);
  /** Track the current blob URL so we can revoke it on unmount or replacement. */
  const previewUrlRef = useRef<string | null>(null);

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

  const checkHealth = useCallback(async () => {
    const requestId = ++healthRequestRef.current;
    setHealthChecking(true);
    setHealthError(null);
    setPipelineReady(null);

    try {
      const health = await healthCheck();
      if (requestId === healthRequestRef.current) {
        setPipelineReady(health.pipeline_ready);
      }
    } catch (err) {
      if (requestId !== healthRequestRef.current) {
        return;
      }
      setPipelineReady(false);
      setHealthError(err instanceof Error ? err.message : "Health check failed");
    } finally {
      if (requestId === healthRequestRef.current) {
        setHealthChecking(false);
      }
    }
  }, []);

  useEffect(() => {
    const requestId = ++healthRequestRef.current;
    void healthCheck()
      .then((health) => {
        if (requestId === healthRequestRef.current) {
          setPipelineReady(health.pipeline_ready);
        }
      })
      .catch((err: unknown) => {
        if (requestId === healthRequestRef.current) {
          setPipelineReady(false);
          setHealthError(
            err instanceof Error ? err.message : "Health check failed"
          );
        }
      })
      .finally(() => {
        if (requestId === healthRequestRef.current) {
          setHealthChecking(false);
        }
      });
    return () => {
      if (healthRequestRef.current === requestId) {
        healthRequestRef.current += 1;
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      predictionRequestRef.current += 1;
    };
  }, []);

  /**
   * Handle file selection from the file input.
   */
  function onFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    if (!selected) {
      return;
    }

    const acceptedTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (!acceptedTypes.includes(selected.type)) {
      setError(
        `Unsupported file type: ${selected.type}. Please upload a JPG or PNG image.`
      );
      setFile(null);
      setPreview(null);
      return;
    }

    if (selected.size > 10 * 1024 * 1024) {
      setError(
        `File too large: ${(selected.size / 1024 / 1024).toFixed(2)} MB. Maximum is 10 MB.`
      );
      setFile(null);
      setPreview(null);
      return;
    }

    // Revoke the previously painted object URL so we do not leak blobs.
    if (previewUrlRef.current && previewUrlRef.current !== preview) {
      URL.revokeObjectURL(previewUrlRef.current);
    }

    previewUrlRef.current = URL.createObjectURL(selected);
    setFile(selected);
    setError(null);
    setResult(null);
    setPreview(previewUrlRef.current);
  }

  /**
   * Trigger file picker.
   */
  function onUploadClick() {
    fileInputRef.current?.click();
  }

  /**
   * Submit the selected file to the backend.
   */
  async function onPredict() {
    if (!file) {
      return;
    }

    setLoading(true);
    setError(null);
    setPredictionStage("Sending image to the backend...");
    const requestId = ++predictionRequestRef.current;

    try {
      const data = await predict(
        file,
        (_progress, stage) => {
          if (requestId === predictionRequestRef.current) {
            setPredictionStage(stage);
          }
        }
      );
      if (requestId === predictionRequestRef.current) {
        setResult(data);
      }
    } catch (err) {
      if (requestId === predictionRequestRef.current) {
        setError(err instanceof Error ? err.message : "Prediction failed");
        setResult(null);
      }
    } finally {
      if (requestId === predictionRequestRef.current) {
        setLoading(false);
      }
    }
  }

  /**
   * Remove the current file and preview.
   */
  function onRemove() {
    setFile(null);
    setResult(null);
    setError(null);
    setPredictionStage("");
    predictionRequestRef.current += 1;
    setLoading(false);

    const current = previewUrlRef.current;
    if (current) {
      previewUrlRef.current = null;
      setPreview(null);
      // Defer revocation so the img element has unloaded the src.
      requestAnimationFrame(() => URL.revokeObjectURL(current));
    }
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
      <div className="connectivity">
        {pipelineReady === null ? (
          <span className="connectivity-status connectivity-status--checking">
            {healthChecking
              ? "Checking backend..."
              : "Backend status not checked"}
          </span>
        ) : pipelineReady ? (
          <span className="connectivity-status connectivity-status--ok">
            Backend ready
          </span>
        ) : (
          <div className="connectivity-status connectivity-status--error">
            <span>
              Backend unavailable
              {healthError ? `: ${healthError}` : "."}
            </span>
            <button
              type="button"
              className="retry-button"
              onClick={() => void checkHealth()}
              disabled={healthChecking}
            >
              {healthChecking ? "Retrying..." : "Retry health check"}
            </button>
          </div>
        )}
      </div>

      {/* Upload area */}
      <section className="upload-section">
        <div
          className={`dropzone ${file ? "dropzone--filled" : ""}`}
          onClick={onUploadClick}
          role="button"
          tabIndex={0}
        >
          {preview ? (
            <div className="preview-wrapper">
              {/* Native <img> is intentional for local blob URL previews. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img className="preview-image" src={preview} alt="Uploaded X-ray" />
              <button
                className="remove-button"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove();
                }}
                type="button"
              >
                Remove
              </button>
            </div>
          ) : (
            <>
              <span className="dropzone-icon">📤</span>
              <span className="dropzone-text">
                Click or drag a chest X-ray image here
              </span>
              <span className="dropzone-hint">
                JPG, JPEG, or PNG — up to 10 MB
              </span>
            </>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            onChange={onFileChange}
            className="file-input"
            disabled={loading}
          />
        </div>

        {error && <div className="error-banner">{error}</div>}

        {file && !loading && !result && (
          <button
            className="predict-button"
            onClick={onPredict}
            disabled={pipelineReady === false}
          >
            Analyze Image
          </button>
        )}

        {loading && (
          <>
            <button
              className="predict-button predict-button--disabled"
              disabled
            >
              Analyzing...
            </button>
            <IndeterminateProgress
              label={predictionStage || "Processing image..."}
            />
          </>
        )}
      </section>

      <aside className="privacy-notice" aria-label="Privacy and data transfer">
        <h2 className="privacy-title">Privacy and data transfer</h2>
        <p>
          When you choose <strong>Analyze Image</strong>, the selected image is
          sent from your browser to this application&apos;s configured backend.
          The production hosted-model path may then process it with a
          third-party model provider (such as Hugging Face). Do not upload
          patient-identifiable images or protected health information.
        </p>
        <p>
          The backend is intended to process the request without persistently
          storing the image or prediction. The image is not sent to a third
          party by the browser, but the provider&apos;s retention and processing
          terms still apply. Remove the image and avoid using this demo for
          clinical decisions.
        </p>
      </aside>

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
