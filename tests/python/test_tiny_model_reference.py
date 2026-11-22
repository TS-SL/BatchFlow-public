"""Independent numerical reference checks for tiny_deterministic.onnx."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

ROOT = Path(__file__).resolve().parents[2]
META = json.loads((ROOT / "models" / "tiny_deterministic.json").read_text(encoding="utf-8"))
MODEL = ROOT / META["path"]
FIXTURES = ROOT / META["fixture"]
RTOL = META["tolerance"]["rtol"]
ATOL = META["tolerance"]["atol"]


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def data():
    return np.load(FIXTURES)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_fixture_parity(session, data, batch):
    x = data[f"x_b{batch}"]
    y_ref = data[f"y_b{batch}"]
    y = session.run(["output"], {"input": x})[0]
    assert y.shape == (batch, 4)
    np.testing.assert_allclose(y, y_ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("batch", [2, 4])
def test_batched_equals_unbatched(session, data, batch):
    x = data[f"x_b{batch}"]
    y_batch = session.run(["output"], {"input": x})[0]
    parts = [session.run(["output"], {"input": x[i : i + 1]})[0] for i in range(batch)]
    y_cat = np.concatenate(parts, axis=0)
    np.testing.assert_allclose(y_batch, y_cat, rtol=RTOL, atol=ATOL)


def test_sha256_matches_meta():
    import hashlib

    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert digest == META["sha256"]
