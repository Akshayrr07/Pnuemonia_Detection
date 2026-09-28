"""Model loading isolated to the evaluation entry point.

The shared deployment/inference loader is intentionally not changed. Evaluation
runs need the selected model order and checkpoint directory to be explicit, so
they get a small loader local to this package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


def load_evaluation_models(
    device,
    model_names: Sequence[str],
    checkpoint_dir: os.PathLike[str] | str,
):
    """Load named checkpoints in the exact order supplied by the evaluator."""

    import torch
    from src.models.model_factory import get_model

    directory = Path(checkpoint_dir)
    models = []
    for name in model_names:
        model = get_model(name, freeze=False).to(device)
        checkpoint_path = directory / f"{name}.pt"
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models


__all__ = ["load_evaluation_models"]
