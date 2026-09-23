"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { healthCheck, predict } from "./lib/api";
import type { PredictResponse } from "./lib/types";

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
  return (
    <div className="progress">
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${value}%` }} />
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

      <div className="disclaimer">
        <span className="disclaimer-icon">⚠️</span>
        <p>{data.disclaimer}</p>
      </div>
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
  const [error, setError] = useState<string | null>(null);
  const [pipelineReady, setPipelineReady] = useState<boolean | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);

  /**
   * Revoke the previous object URL whenever the preview changes or the
   * component unmounts, so we do not leak blob URLs.
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
   * Track the preview URL so onRemove() can revoke the correct one.
   */
  useEffect(() => {
    previewUrlRef.current = preview;
    return () => {
      // When preview changes away from this value, revoke the old one.
      if (previewUrlRef.current === preview && preview) {
        URL.revokeObjectURL(preview);
        previewUrlRef.current = null;
      }
    };
  }, [preview]);

  /**
   * Report progress from the predict() call back into our label state.
   */
  const onProgress = useCallback(
    (value: number, stage: string) => {
      setProgress(value);
      setProgressLabel(stage);
    },
    []
  );

  /**
   * Check backend health on mount.
   */
  useEffect(() => {
    healthCheck()
      .then((h) => setPipelineReady(h.pipeline_ready))
      .catch(() => setPipelineReady(false));
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
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }

    setFile(selected);
    setError(null);
    setResult(null);
    setPreview(URL.createObjectURL(selected));
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
    setProgress(0);
    setProgressLabel("Starting...");

    try {
      const data = await predict(file, onProgress);
      setResult(data);
      setProgress(100);
      setProgressLabel("Complete");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  /**
   * Remove the current file and preview.
   */
  function onRemove() {
    setFile(null);
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreview(null);
    setResult(null);
    setError(null);
    setProgress(0);
    setProgressLabel("");
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
      <section className="upload-section">
        <div
          className={`dropzone ${file ? "dropzone--filled" : ""}`}
          onClick={onUploadClick}
          role="button"
          tabIndex={0}
        >
          {preview ? (
            <div className="preview-wrapper">
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
            <ProgressBar value={progress} label={progressLabel} />
          </>
        )}
      </section>

      {/* Result */}
      {result && <ResultCard data={result} />}

      {/* Footer */}
      <footer className="footer">
        <p>
          Powered by a hierarchical classification pipeline: Normal vs Pneumonia,
          then Bacterial vs Viral subtype.
        </p>
      </footer>
    </main>
  );
}
