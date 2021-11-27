from unittest import TestCase
from dataclasses import dataclass

from emo_classifier.artifact import JsonArtifact
from lib import PROJ_ROOT

@dataclass
class DummyJsonArtifact(JsonArtifact):
    name: str
    value: int

class TestJsonArtifact(TestCase):
    dummy_name = "dummy_name"
    dummy_value = 1234
    file_path = PROJ_ROOT / "emo_classifier/data/DummyJsonArtifact.json"

    @classmethod
    def setUpClass(cls) -> None:
        if cls.file_path.exists():
            cls.file_path.unlink(missing_ok=True)

    def test_save_and_load(self):
        dummy_json_artifact = DummyJsonArtifact(self.dummy_name, self.dummy_value)
        dummy_json_artifact.save()
        self.assertTrue(self.file_path.exists())

        loaded_artifact = DummyJsonArtifact.load()
        self.assertIsInstance(loaded_artifact, DummyJsonArtifact)
        self.assertEqual(dummy_json_artifact, loaded_artifact)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.file_path.unlink(missing_ok=True)
