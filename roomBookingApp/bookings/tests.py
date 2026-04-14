import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Face_Recognition.model_state import (
	build_dataset_fingerprint,
	dataset_requires_retraining,
	load_training_state,
	save_training_state,
)


class FaceDatasetStateTests(SimpleTestCase):
	def test_fingerprint_changes_when_new_image_is_added(self):
		with TemporaryDirectory() as temp_dir:
			dataset_dir = Path(temp_dir) / "dataset"
			person_dir = dataset_dir / "alice"
			person_dir.mkdir(parents=True)

			(person_dir / "1.jpg").write_bytes(b"image-one")
			first_signature = build_dataset_fingerprint(dataset_dir)["dataset_signature"]

			(person_dir / "2.jpg").write_bytes(b"image-two")
			second_signature = build_dataset_fingerprint(dataset_dir)["dataset_signature"]

			self.assertNotEqual(first_signature, second_signature)

	def test_non_image_files_are_ignored(self):
		with TemporaryDirectory() as temp_dir:
			dataset_dir = Path(temp_dir) / "dataset"
			person_dir = dataset_dir / "bob"
			person_dir.mkdir(parents=True)

			(person_dir / "1.jpg").write_bytes(b"image-one")
			baseline = build_dataset_fingerprint(dataset_dir)["dataset_signature"]

			(person_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
			after_text_file = build_dataset_fingerprint(dataset_dir)["dataset_signature"]

			self.assertEqual(baseline, after_text_file)

	def test_saved_state_controls_retraining_decision(self):
		with TemporaryDirectory() as temp_dir:
			dataset_dir = Path(temp_dir) / "dataset"
			state_path = Path(temp_dir) / "face_model_state.json"
			person_dir = dataset_dir / "carol"
			person_dir.mkdir(parents=True)
			(person_dir / "1.jpg").write_bytes(b"image-one")

			current_state = build_dataset_fingerprint(dataset_dir)
			save_training_state(state_path, current_state)

			self.assertEqual(load_training_state(state_path)["dataset_signature"], current_state["dataset_signature"])
			self.assertFalse(dataset_requires_retraining(dataset_dir, state_path))

			(person_dir / "2.jpg").write_bytes(b"image-two")
			self.assertTrue(dataset_requires_retraining(dataset_dir, state_path))
