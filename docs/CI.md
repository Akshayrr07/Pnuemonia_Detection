# Continuous integration and pull-request automation

This repository uses GitHub Actions for pull-request validation and automatic
squash merging after the required checks pass.

## Triggers

The `CI` workflow runs for:

- every pull request;
- pushes to `main`;
- manual runs from the Actions tab.

The `Auto-merge` workflow is triggered by the `CI` workflow's `completed` event.
It only starts when the entire CI run has concluded `success`, then finds the
open pull request for that exact commit and enables GitHub auto-merge.
Pull requests from forks still run CI, but auto-merge is skipped.

## Checks

### Backend / Python

- Python 3.11 syntax compilation.
- Ruff linting.
- Mypy type checking for the FastAPI backend and inference package.
- The repository's executable backend verification scripts.
- `pip check` dependency consistency.
- `pip-audit` against `backend/requirements.txt`.

### Frontend / Next.js

- `npm ci` for deterministic dependency installation.
- ESLint with zero warnings allowed.
- `next typegen` followed by `tsc --noEmit`.
- Next.js production build.
- `npm audit` at the high-severity threshold.

There is currently no separate frontend unit-test suite, so the frontend gate
consists of lint, type checking, build, and dependency audit.

### Backend / Docker build

- Builds the production backend image.
- Starts the image and verifies `GET /health` responds successfully.

The required status-check names are:

- `Backend / Python`
- `Backend / Verification tests`
- `Frontend / Next.js`
- `Backend / Docker build`

## One-time GitHub setup

1. Add a repository secret named `CI_ADMIN_TOKEN`. It must be a fine-grained
   token for this repository with **Administration: Read and write**. The
   default `GITHUB_TOKEN` cannot change branch protection settings.
2. In the repository settings, enable **Allow auto-merge**.
3. Run the **Branch protection setup** workflow from the Actions tab once.
   It protects `main`, requires all four checks above, requires up-to-date
   branches, requires linear history, blocks force pushes and deletion, and
   resolves conversations before merging.
4. Open future pull requests from branches in this repository. The
   `Auto-merge` workflow enables GitHub auto-merge and uses a squash merge once
   the required checks are green.

The branch-protection workflow is intentionally manual: it needs an
administrative token and should be run deliberately after these workflow files
are present on `main`.
