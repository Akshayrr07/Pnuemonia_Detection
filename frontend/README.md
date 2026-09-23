# Pneumonia Detection — Web Frontend

Next.js web application for the Pneumonia Detection system. This is the user-facing interface for uploading chest X-rays and viewing hierarchical predictions.

**Live deployment:** `https://pneumonia-detection-8fz.pages.dev`

## What it does

- Upload a chest X-ray (JPEG or PNG, up to 10 MB).
- Preview the uploaded image.
- Submit it to the backend and view the result:
  - Primary prediction (Normal or Pneumonia).
  - Confidence score.
  - Subtype (Bacterial or Viral Pneumonia) — shown only when pneumonia is detected.
  - Per-class probability breakdown.
  - Medical disclaimer and limitation notice.
- Optional Grad-CAM heatmap overlay (only when the backend is configured with a local model checkpoint).

## Tech stack

- **Framework:** Next.js 16, TypeScript, App Router.
- **Hosting:** Cloudflare Pages static export (`output: "export"` in `next.config.ts`).
- **Styling:** plain CSS (no Tailwind).
- **Communication with backend:** `frontend/app/lib/api.ts` calls the FastAPI backend via `NEXT_PUBLIC_BACKEND_URL`. When that env var is unset, the frontend falls back to same-origin, which works when the backend is served from the same domain.

## Development

From the `frontend/` directory:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. The dev server does server-side rendering and is not the same as the static export used for production.

## Static export (production build)

The production build is a static export for Cloudflare Pages:

```bash
npm run build
```

This produces a static site in `out/` with pages for `/`, `/research`, and the 404 fallback. The Cloudflare Pages deployment uploads the contents of `out/`.

`next.config.ts` is configured for this:

```ts
output: "export",
trailingSlash: true,
images: { unoptimized: true },
```

Images are left unoptimized because Cloudflare Pages serves the exported files directly and does not run the Next.js image optimization pipeline.

## Environment variables

| Variable | Purpose |
|---|---|
| `NEXT_PUBLIC_BACKEND_URL` | URL of the deployed backend API. When unset, the frontend calls the same origin. |

Set this in the Cloudflare Pages project settings (or in your local `.env`) to point at the deployed backend. For example:

```
NEXT_PUBLIC_BACKEND_URL=https://your-backend.example.com
```

## Pages

- **Homepage (`/`)** — upload UI, result card, disclaimer, heatmap toggle.
- **Research page (`/research`)** — dataset summary, split strategy, model families, and research results (accuracy only, sourced from `docs/RESEARCH_SUMMARY.md`).

## Relationship to the rest of the system

- The frontend calls the backend at `NEXT_PUBLIC_BACKEND_URL`. It does not call Hugging Face directly.
- The backend is documented in `backend/README.md`.
- The full system is summarized in `docs/FINAL_SUMMARY.md`.
- Deployment design is in `docs/DEPLOYMENT.md`.

## Notes

- The research page reports accuracy numbers from the research phase. These are not live inference results and may differ from what the deployed backend returns if the hosted checkpoints differ from the ones used in research.
- Grad-CAM is enabled on the frontend only when the backend returns a non-null `heatmap_b64`. The backend only produces that field when `LOCAL_MODEL_PATH` points to a real checkpoint.
