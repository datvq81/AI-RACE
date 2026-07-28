"""Tests for rejecting partial Nerfstudio checkpoints during resume."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.run_local_validation import (
    _checkpoint_archive_is_readable,
    _checkpoint_step,
    _complete_configs,
)


class CheckpointResumeTest(unittest.TestCase):
    def test_parses_nerfstudio_checkpoint_step(self) -> None:
        self.assertEqual(_checkpoint_step(Path("step-000019999.ckpt")), 19999)
        self.assertIsNone(_checkpoint_step(Path("latest.ckpt")))

    def test_accepts_structural_pytorch_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "step-000019999.ckpt"
            with zipfile.ZipFile(checkpoint, "w") as archive:
                archive.writestr("archive/data.pkl", b"x")
                archive.writestr("archive/version", b"3")
            readable, reason = _checkpoint_archive_is_readable(checkpoint)
            self.assertTrue(readable, reason)

    def test_rejects_truncated_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "step-000019999.ckpt"
            checkpoint.write_bytes(b"PK\x03\x04truncated")
            readable, _ = _checkpoint_archive_is_readable(checkpoint)
            self.assertFalse(readable)

    def test_complete_configs_requires_final_readable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            run_dir = project_root / "outputs" / "experiment" / "method" / "run"
            model_dir = run_dir / "nerfstudio_models"
            model_dir.mkdir(parents=True)
            config = run_dir / "config.yml"
            config.write_text("test: true\n", encoding="utf-8")

            partial = model_dir / "step-000017999.ckpt"
            with zipfile.ZipFile(partial, "w") as archive:
                archive.writestr("archive/data.pkl", b"x")
            with patch(
                "scripts.run_local_validation.PROJECT_ROOT",
                project_root,
            ):
                self.assertEqual(_complete_configs("experiment", 19999), [])

            partial.rename(model_dir / "step-000019999.ckpt")
            with patch(
                "scripts.run_local_validation.PROJECT_ROOT",
                project_root,
            ):
                self.assertEqual(
                    _complete_configs("experiment", 19999),
                    [config],
                )


if __name__ == "__main__":
    unittest.main()
