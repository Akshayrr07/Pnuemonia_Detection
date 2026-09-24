import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from PIL import Image
from torch import nn

import src.inference.explainability as explainability
from src.inference.explainability import generate_heatmap, grad_cam_class_index
from src.models.model_factory import get_model


def test_grad_cam_class_index_maps_concrete_three_class_labels():
    assert grad_cam_class_index("Normal") == 0
    assert grad_cam_class_index("Bacterial Pneumonia") == 1
    assert grad_cam_class_index("Viral Pneumonia") == 2


def test_grad_cam_class_index_rejects_aggregate_pneumonia():
    with pytest.raises(ValueError, match="aggregate Pneumonia"):
        grad_cam_class_index("Pneumonia")


class TinyGradCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.classifier = nn.Linear(4 * 8 * 8, 3)

    def forward(self, inputs):
        features = self.features(inputs)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (8, 8))
        return self.classifier(pooled.flatten(start_dim=1))


def test_generate_heatmap_returns_data_url_for_real_supported_model():
    model = get_model("resnet", num_classes=3)
    image = Image.new("RGB", (96, 64), color=(32, 64, 96))

    data_url = generate_heatmap(
        model=model,
        model_name="resnet",
        image=image,
        class_idx=1,
        device=torch.device("cpu"),
    )

    assert data_url is not None
    assert data_url.startswith("data:image/png;base64,")

    png_bytes = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    assert png_bytes
    with Image.open(io.BytesIO(png_bytes)) as heatmap:
        assert heatmap.format == "PNG"
        assert heatmap.size == image.size
    assert model.training


def test_generate_heatmap_uses_declared_lock_and_returns_none_on_failure(monkeypatch):
    image = Image.new("RGB", (32, 32), color=(16, 32, 48))
    lock_entered = threading.Event()

    class RecordingLock:
        def __enter__(self):
            lock_entered.set()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(explainability, "_LOCK", RecordingLock())

    result = generate_heatmap(
        model=TinyGradCamModel(),
        model_name="unsupported",
        image=image,
        class_idx=0,
        device=torch.device("cpu"),
    )

    assert result is None
    assert lock_entered.is_set()


def test_generate_heatmap_serializes_model_and_gradient_operations(monkeypatch):
    model = TinyGradCamModel()
    image = Image.new("RGB", (48, 48), color=(24, 48, 72))
    operations_in_progress = 0
    max_operations_in_progress = 0
    operations_lock = threading.Lock()
    first_call_started = threading.Event()
    release_first_call = threading.Event()

    class ConcurrentModel(nn.Module):
        def forward(self, inputs):
            nonlocal operations_in_progress, max_operations_in_progress
            with operations_lock:
                operations_in_progress += 1
                max_operations_in_progress = max(
                    max_operations_in_progress,
                    operations_in_progress,
                )
            try:
                first_call_started.set()
                release_first_call.wait(timeout=0.2)
                features = model.features(inputs)
                pooled = torch.nn.functional.adaptive_avg_pool2d(features, (8, 8))
                return model.classifier(pooled.flatten(start_dim=1))
            finally:
                with operations_lock:
                    operations_in_progress -= 1

    slow_model = ConcurrentModel()
    slow_model.features = model.features
    monkeypatch.setattr(explainability, "resolve_target_layer", lambda current, name: current.features)

    def run_heatmap():
        return generate_heatmap(
            model=slow_model,
            model_name="tiny",
            image=image,
            class_idx=1,
            device=torch.device("cpu"),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_heatmap)
        assert first_call_started.wait(timeout=1)
        second = executor.submit(run_heatmap)
        release_first_call.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert all(result and result.startswith("data:image/png;base64,") for result in results)
    assert max_operations_in_progress == 1
