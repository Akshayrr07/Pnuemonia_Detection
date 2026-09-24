"""Regression tests for hardened image uploads."""

from __future__ import annotations

import io
import os
import warnings
from collections.abc import Generator
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import Headers

os.environ.setdefault("HF_BINARY_MODEL_ID", "test/binary")
os.environ.setdefault("HF_SUBTYPE_MODEL_ID", "test/subtype")

import backend.app as backend_module


@pytest.fixture(autouse=True)
def _keep_pipeline_state() -> Generator[None, None, None]:
    """Do not let a validation-only test affect other backend tests."""
    yield
    backend_module._pipeline = None


def image_bytes(
    image_format: str,
    *,
    size: tuple[int, int] = (32, 24),
) -> bytes:
    image = Image.new("RGB", size, color=(10, 20, 30))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def upload(contents: bytes, *, filename: str, content_type: str) -> UploadFile:
    headers = Headers({"content-type": content_type})
    return UploadFile(file=io.BytesIO(contents), filename=filename, headers=headers)


def assert_rejected(image: UploadFile, expected_detail: str) -> None:
    with pytest.raises(HTTPException) as caught:
        backend_module._validate_upload(image)
    assert caught.value.status_code == 400
    assert expected_detail in caught.value.detail


def test_rejects_content_beyond_configured_limit_without_unbounded_read() -> None:
    max_upload_bytes = backend_module.UPLOAD_MAX_SIZE_MB * 1024 * 1024
    source = io.BytesIO(b"x" * (max_upload_bytes + 1))
    image = UploadFile(
        file=source,
        filename="xray.png",
        headers=Headers({"content-type": "image/png"}),
    )

    reads: list[int | None] = []

    class GuardedSource:
        def read(self, size: int = -1) -> bytes:
            reads.append(size)
            assert size >= 0
            return source.read(size)

    image.file = GuardedSource()  # type: ignore[assignment]

    assert_rejected(image, "Image too large")

    chunk_size = 1024 * 1024
    full_chunks, remainder = divmod(max_upload_bytes + 1, chunk_size)
    expected_reads = [chunk_size] * full_chunks
    if remainder:
        expected_reads.append(remainder)
    assert reads == expected_reads
    assert source.tell() == max_upload_bytes + 1


def test_rejects_exactly_configured_limit_plus_one_without_full_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_module, "UPLOAD_MAX_SIZE_MB", 1)
    source = io.BytesIO(b"x" * (1024 * 1024 + 1))
    image = UploadFile(
        file=source,
        filename="xray.png",
        headers=Headers({"content-type": "image/png"}),
    )

    assert_rejected(image, "Image too large")


def test_rejects_dimensions_and_pixel_count_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = image_bytes("PNG", size=(64, 64))

    def fail_if_loaded(image: Image.Image) -> None:
        raise AssertionError("decoded before checks")

    monkeypatch.setattr(backend_module.Image.Image, "load", fail_if_loaded)
    monkeypatch.setattr(backend_module, "MAX_IMAGE_WIDTH", 32)
    monkeypatch.setattr(backend_module, "MAX_IMAGE_HEIGHT", 32)
    monkeypatch.setattr(backend_module, "MAX_IMAGE_PIXELS", 32 * 32)

    assert_rejected(
        upload(contents, filename="xray.png", content_type="image/png"),
        "Image dimensions exceed limit",
    )

    # Width, height, and total pixels are independent limits.
    monkeypatch.setattr(backend_module, "MAX_IMAGE_WIDTH", 128)
    assert_rejected(
        upload(contents, filename="xray.png", content_type="image/png"),
        "Image dimensions exceed limit",
    )

    monkeypatch.setattr(backend_module, "MAX_IMAGE_HEIGHT", 128)
    monkeypatch.setattr(backend_module, "MAX_IMAGE_PIXELS", 32 * 32)
    assert_rejected(
        upload(contents, filename="xray.png", content_type="image/png"),
        "Image dimensions exceed limit",
    )


def test_rejects_gif_renamed_as_png() -> None:
    image = upload(
        image_bytes("GIF"),
        filename="xray.png",
        content_type="image/png",
    )

    assert_rejected(image, "Unsupported image format: GIF")


def test_rejects_multi_frame_images() -> None:
    output = io.BytesIO()
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(
        output,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (16, 16), color=(4, 5, 6))],
    )
    image = upload(output.getvalue(), filename="xray.png", content_type="image/png")

    assert_rejected(image, "Multi-frame images are not supported")


def test_rejects_pillow_decompression_bomb_warning() -> None:
    image = upload(
        image_bytes("PNG", size=(32, 32)),
        filename="xray.png",
        content_type="image/png",
    )

    original_open = Image.open

    def open_with_bomb_warning(*args: object, **kwargs: object) -> Image.Image:
        warnings.warn("simulated decompression bomb warning", Image.DecompressionBombWarning)
        return original_open(args[0])  # type: ignore[arg-type]

    with patch.object(backend_module.Image, "open", side_effect=open_with_bomb_warning):
        assert_rejected(image, "Could not decode image")


def test_rejects_pillow_decompression_bomb_error() -> None:
    image = upload(
        image_bytes("PNG", size=(32, 32)),
        filename="xray.png",
        content_type="image/png",
    )

    def open_with_bomb_error(*args: object, **kwargs: object) -> Image.Image:
        raise Image.DecompressionBombError("simulated decompression bomb error")

    with patch.object(backend_module.Image, "open", side_effect=open_with_bomb_error):
        assert_rejected(image, "Could not decode image")


def test_rejects_real_pillow_decompression_bomb_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = upload(
        image_bytes("PNG", size=(32, 32)),
        filename="xray.png",
        content_type="image/png",
    )
    monkeypatch.setattr(backend_module.Image, "MAX_IMAGE_PIXELS", 100)

    assert_rejected(image, "Could not decode image")


@pytest.mark.parametrize(
    ("image_format", "filename", "content_type"),
    [
        ("PNG", "xray.png", "image/png"),
        ("JPEG", "xray.jpg", "image/jpeg"),
    ],
)
def test_accepts_valid_png_and_jpeg(
    image_format: str,
    filename: str,
    content_type: str,
) -> None:
    image = upload(
        image_bytes(image_format),
        filename=filename,
        content_type=content_type,
    )

    validated = backend_module._validate_upload(image)

    assert validated.mode == "RGB"
    assert validated.size == (32, 24)


def test_predict_keeps_valid_png_status_code() -> None:
    from src.inference.schemas import HierarchicalPrediction

    prediction = HierarchicalPrediction(
        primary_prediction="Normal",
        primary_confidence=0.99,
        subtype_prediction=None,
        subtype_confidence=None,
        probabilities={"Normal": 0.99, "Pneumonia": 0.01},
        model_outputs={"binary": {"label": "Normal"}},
    )

    class FakePipeline:
        def predict(self, image: Image.Image) -> HierarchicalPrediction:
            assert image.mode == "RGB"
            return prediction

    # Use a bare client so lifespan does not replace the fake pipeline.
    with TestClient(backend_module.app) as client:
        backend_module._pipeline = FakePipeline()
        response = client.post(
            "/predict",
            files={"file": ("xray.png", image_bytes("PNG"), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["primary_prediction"] == "Normal"
