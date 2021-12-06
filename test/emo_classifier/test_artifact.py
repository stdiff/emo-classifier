from dataclasses import dataclass

from emo_classifier.artifact import JsonArtifact
from lib import PROJ_ROOT

dummy_name = "dummy_name"
dummy_value = 1234
file_path = PROJ_ROOT / "emo_classifier/data/DummyJsonArtifact.json"

if file_path.exists():
    file_path.unlink(missing_ok=True)


@dataclass
class DummyJsonArtifact(JsonArtifact):
    name: str
    value: int


def test_save_and_load():
    dummy_json_artifact = DummyJsonArtifact(dummy_name, dummy_value)
    dummy_json_artifact.save()
    assert file_path.exists()

    loaded_artifact = DummyJsonArtifact.load()
    assert isinstance(loaded_artifact, DummyJsonArtifact)
    assert dummy_json_artifact == loaded_artifact

    file_path.unlink(missing_ok=True)
