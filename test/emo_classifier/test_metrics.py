from dataclasses import dataclass
from pathlib import Path

import pytest

from emo_classifier.metrics import JsonArtifact
from training import LocalPaths

dummy_name = "dummy_name"
dummy_value = 1234


@pytest.fixture(scope="module")
def file_path():
    local_paths = LocalPaths()
    file_path = local_paths.project_root / "emo_classifier/resources/DummyJsonArtifact.json"
    file_path.unlink(missing_ok=True)
    yield file_path
    file_path.unlink(missing_ok=True)


@dataclass
class DummyJsonArtifact(JsonArtifact):
    name: str
    value: int


def test_save_and_load(file_path: Path):
    dummy_json_artifact = DummyJsonArtifact(dummy_name, dummy_value)
    dummy_json_artifact.save()
    assert file_path.exists()

    loaded_artifact = DummyJsonArtifact.load()
    assert isinstance(loaded_artifact, DummyJsonArtifact)
    assert dummy_json_artifact == loaded_artifact
