# Continuous integration and pull-request automation

This repository uses GitHub Actions for pull-request validation and automatic
squash merging after the required checks pass.

## Triggers

The `CI` workflow runs for:

- every pull request;
- pushes to any branch;
- manual runs from the Actions tab.

The `Auto-merge` workflow is triggered when either the `CI` or the separate
`Frontend clean build` workflow completes. It only starts when that workflow
has concluded `success`, then waits until all required checks for the same PR
commit are green before enabling GitHub auto-merge.
Pull requests from forks still run CI, but auto-merge is skipped.

## Checks

### Python (compile, phase tests, dependency audit)

- Python source compilation.
- The repository's executable backend verification scripts.
- `pip-audit` against the pinned backend and research dependency metadata.

### Frontend (clean install, lint, typecheck, build)

- `npm ci` for deterministic dependency installation.
- ESLint with zero warnings allowed.
- `next typegen` followed by `tsc --noEmit`.
- Next.js production build.
- `npm audit` at the high-severity threshold.

There is currently no separate frontend unit-test suite, so the frontend gate
consists of lint, type checking, build, and dependency audit.

### Docker image

- Builds the production backend image.
- Starts the image and verifies `GET /health` responds successfully.

The required status-check names are:

- `Frontend (clean install, lint, typecheck, build)`
- `Python (compile, phase tests, dependency audit)`
- `Docker image`
- `frontend` (the separate clean-build workflow)

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
