"""Model loading isolated to the evaluation entry point.

Evaluation runs need the selected model order and checkpoint directory to be
explicit, so the orchestration stays local to this package.  Checkpoint
deserialization itself is delegated to the shared trusted loader so evaluation
cannot drift from the deployment/inference safety policy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from src.inference.checkpoint import load_trusted_checkpoint


def load_evaluation_models(
    device,
    model_names: Sequence[str],
    checkpoint_dir: os.PathLike[str] | str,
):
    """Load named checkpoints in the exact order supplied by the evaluator."""

    from src.models.model_factory import get_model

    directory = Path(checkpoint_dir)
    models = []
    for name in model_names:
        model = get_model(name, freeze=False).to(device)
        checkpoint_path = directory / f"{name}.pt"
        state_dict = load_trusted_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        models.append(model)
    return models


__all__ = ["load_evaluation_models"]
