import Link from "next/link";

/**
 * Two-column metric card for a classification stage.
 */
function MetricCard({
  title,
  subtitle,
  accuracy,
  size,
}: {
  title: string;
  subtitle: string;
  accuracy: number | string;
  size?: "sm" | "lg";
}) {
  return (
    <div
      className={`metric-card metric-card--${size ?? "lg"}`}
      style={{ "--accent": "var(--accent)" } as React.CSSProperties}
    >
      <div className="metric-header">
        <h3 className="metric-title">{title}</h3>
        <span className="metric-subtitle">{subtitle}</span>
      </div>
      <div className="metric-numbers">
        <div className="metric-number">
          <span className="metric-value metric-value--accent">
            {typeof accuracy === "string"
              ? accuracy
              : `${(accuracy * 100).toFixed(2)}%`}
          </span>
          <span className="metric-label">Accuracy</span>
        </div>
        <div className="metric-number metric-number--pending">
          <span className="metric-value metric-value--muted">—</span>
          <span className="metric-label">Precision</span>
        </div>
        <div className="metric-number metric-number--pending">
          <span className="metric-value metric-value--muted">—</span>
          <span className="metric-label">Recall</span>
        </div>
        <div className="metric-number metric-number--pending">
          <span className="metric-value metric-value--muted">—</span>
          <span className="metric-label">F1 Score</span>
        </div>
      </div>
      <p className="metric-note">
        Per-class precision, recall, and F1 are pending final evaluation.
      </p>
    </div>
  );
}

/**
 * Placeholder confusion matrix block with a caption.
 */
function ConfusionMatrixPlaceholder({
  title,
  shape,
}: {
  title: string;
  shape: "square" | "wide";
}) {
  return (
    <div className="cm-placeholder">
      <span className="cm-caption">{title}</span>
      <div className={`cm-svg cm-svg--${shape}`}>
        <svg
          viewBox="0 0 200 160"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="cm-svg-inner"
        >
          {/* grid */}
          <rect x="10" y="10" width="180" height="140" rx="6" fill="#f3f4f6" />
          <line x1="100" y1="10" x2="100" y2="150" stroke="#d1d5db" strokeWidth="1" />
          <line x1="10" y1="80" x2="190" y2="80" stroke="#d1d5db" strokeWidth="1" />
          {/* labels */}
          <text x="100" y="6" textAnchor="middle" fontSize="9" fill="#6b7280">
            Predicted →
          </text>
          <text
            x="4"
            y="84"
            textAnchor="middle"
            fontSize="9"
            fill="#6b7280"
            transform="rotate(-90 4 84)"
          >
            Actual ↓
          </text>
          {/* cell placeholders */}
          <rect x="20" y="20" width="75" height="55" rx="3" fill="#e5e7eb" />
          <rect x="105" y="20" width="75" height="55" rx="3" fill="#e5e7eb" />
          <rect x="20" y="90" width="75" height="55" rx="3" fill="#e5e7eb" />
          <rect x="105" y="90" width="75" height="55" rx="3" fill="#e5e7eb" />
          {/* cell text placeholders */}
          <text x="57" y="48" textAnchor="middle" fontSize="10" fill="#9ca3af">
            n
          </text>
          <text x="142" y="48" textAnchor="middle" fontSize="10" fill="#9ca3af">
            n
          </text>
          <text x="57" y="118" textAnchor="middle" fontSize="10" fill="#9ca3af">
            n
          </text>
          <text x="142" y="118" textAnchor="middle" fontSize="10" fill="#9ca3af">
            n
          </text>
        </svg>
      </div>
    </div>
  );
}

/**
 * Horizontal flow step for the hierarchical explainer.
 */
function FlowStep({
  number,
  title,
  description,
  color,
}: {
  number: string;
  title: string;
  description: string;
  color: string;
}) {
  return (
    <div className="flow-step">
      <div className="flow-step-badge" style={{ backgroundColor: color }}>
        {number}
      </div>
      <div className="flow-step-body">
        <div className="flow-step-title">{title}</div>
        <div className="flow-step-desc">{description}</div>
      </div>
    </div>
  );
}

export default function ResearchPage() {
  return (
    <main className="page">
      <header className="page-header">
        <div className="page-header-inner">
          <h1 className="page-title">Research Results</h1>
          <p className="page-subtitle">
            Model performance from the classification experiments.
          </p>
          <Link href="/" className="back-link">
            ← Back to detection
          </Link>
        </div>
      </header>

      {/* Dataset summary */}
      <section className="section">
        <h2 className="section-title">Dataset</h2>
        <p className="section-text">
          The study used a controlled dataset of{" "}
          <strong>5,891 unique chest X-rays</strong> assembled from three Kaggle
          sources. Exact duplicates were removed using SHA256 hashing. Labels
          were normalized to three classes:{" "}
          <code>Normal</code>, <code>Bacterial Pneumonia</code>, and{" "}
          <code>Viral Pneumonia</code>.
        </p>
        <p className="section-text">
          The dataset was split into <strong>70% training</strong>,{" "}
          <strong>15% validation</strong>, and <strong>15% test</strong>, with
          stratification by class.
        </p>
      </section>

      {/* Metric cards */}
      <section className="section">
        <h2 className="section-title">Classification Results</h2>
        <div className="metrics-grid">
          <MetricCard
            title="Three-Class Classification"
            subtitle="Normal vs Bacterial vs Viral"
            accuracy="approximately 74–77%"
            size="lg"
          />
          <MetricCard
            title="Binary Pneumonia Detection"
            subtitle="Normal vs Pneumonia"
            accuracy={0.9378}
            size="sm"
          />
          <MetricCard
            title="Pneumonia Subtype Classification"
            subtitle="Bacterial vs Viral"
            accuracy={0.7567}
            size="sm"
          />
        </div>
      </section>

      {/* Confusion matrix placeholders */}
      <section className="section">
        <h2 className="section-title">Confusion Matrices</h2>
        <p className="section-text">
          Confusion matrices are placeholders pending final runs. The matrices
          below show the intended structure; numeric values will be populated
          from the final evaluation outputs.
        </p>
        <div className="cm-grid">
          <ConfusionMatrixPlaceholder
            title="Three-Class Confusion Matrix"
            shape="square"
          />
          <ConfusionMatrixPlaceholder
            title="Binary Pneumonia Confusion Matrix"
            shape="wide"
          />
          <ConfusionMatrixPlaceholder
            title="Subtype Confusion Matrix"
            shape="wide"
          />
        </div>
      </section>

      {/* Why hierarchical */}
      <section className="section">
        <h2 className="section-title">Why Hierarchical Classification?</h2>
        <p className="section-text">
          A single three-class classifier struggled to separate all categories
          in one step, especially the viral class. In contrast, the binary
          pneumonia detector reached <strong>93.78%</strong> accuracy, while
          subtype classification remained harder at about{" "}
          <strong>75.67%</strong>.
        </p>
        <p className="section-text">
          The hierarchical design mirrors these results: first decide whether
          pneumonia is present, then classify the subtype only when needed. This
          keeps the stronger signal (Normal vs Pneumonia) primary and isolates
          the harder decision to cases where it actually matters.
        </p>
        <div className="flow">
          <FlowStep
            number="1"
            title="Detect pneumonia presence"
            description="A binary classifier decides Normal vs Pneumonia. This stage is the most reliable in the pipeline."
            color="var(--accent)"
          />
          <div className="flow-arrow" aria-hidden="true">→</div>
          <FlowStep
            number="2"
            title="Classify subtype when pneumonia is present"
            description="Only if the image is classified as pneumonia does the subtype classifier run, choosing between Bacterial and Viral Pneumonia."
            color="var(--bacterial)"
          />
          <div className="flow-arrow" aria-hidden="true">→</div>
          <FlowStep
            number="3"
            title="Return unified result"
            description="The final response includes the primary prediction, subtype prediction when applicable, confidence scores, and a medical disclaimer."
            color="var(--normal)"
          />
        </div>
      </section>

      {/* Ensemble direction */}
      <section className="section">
        <h2 className="section-title">Ensemble Strategy</h2>
        <p className="section-text">
          The codebase includes soft voting and weighted soft voting utilities
          for combining probabilities across multiple models (MobileNet,
          EfficientNet, ResNet). Ensembles remain a research direction and may
          improve robustness, but the production API exposes a stable
          hierarchical prediction contract regardless of the underlying model
          configuration.
        </p>
        <p className="section-text">
          Future work includes ROC/AUC analysis, sensitivity and specificity
          reporting, calibration analysis, threshold tuning, and external
          validation on unseen datasets.
        </p>
      </section>

      {/* Limitations */}
      <section className="section section--limitations">
        <h2 className="section-title">Limitations</h2>
        <ul className="limitations-list">
          <li>
            The raw datasets are not included in the repository due to size.
          </li>
          <li>
            Split CSVs may contain machine-specific file paths and should be
            regenerated for new environments.
          </li>
          <li>
            Some experiment scripts under{" "}
            <code>experiments/research_scripts/</code> are legacy research
            scripts and should be consolidated before production use.
          </li>
          <li>
            Three-class bacterial-versus-viral classification remains
            challenging.
          </li>
          <li>
            The project is not a certified medical diagnostic system.
          </li>
          <li>
            Confusion matrices and per-class metrics are placeholders pending
            final evaluation runs.
          </li>
        </ul>
      </section>
    </main>
  );
}
