import contextlib
import io
import json
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from fs_iqa.cli import main
from fs_iqa.data import read_image, select_images


class DataTests(unittest.TestCase):
    def test_explicit_manifest_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png", "c.png"):
                Image.new("RGB", (8, 6)).save(root/name)
            manifest = root/"list.txt"
            manifest.write_text("b.png\na.png\n")
            images = select_images(root, image_list=manifest)
            self.assertEqual([p.name for p in images], ["b.png", "a.png"])
            self.assertEqual(tuple(read_image(images[0]).shape), (1, 3, 384, 512))

    def test_dry_run_without_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 6)).save(root/"one.png")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(["--image-dir", str(root), "--dry-run"])
            result = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(result["selected_images"], 1)
            self.assertEqual(len(result["missing_checkpoints"]), 3)

    def test_manifest_outside_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/"list.txt"
            manifest.write_text("../outside.png\n")
            with self.assertRaises(ValueError):
                select_images(root, image_list=manifest)

    def test_cli_end_to_end_with_tiny_model(self):
        import torch
        from torch import nn
        from fs_iqa.models import FSIQA

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 6), (120, 100, 80)).save(root/"one.png")
            placeholder = root/"placeholder.pth"
            placeholder.touch()
            model = FSIQA(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()),
                          nn.Linear(3, 3), nn.Linear(3, 1)).eval()
            model.requires_grad_(False)
            output = root/"outputs"/"smoke.csv"
            with patch("fs_iqa.models.load_model", return_value=model):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    status = main(["--image-dir", str(root), "--mode", "ive",
                                   "--backbone-checkpoint", str(placeholder),
                                   "--ftn-checkpoint", str(placeholder),
                                   "--scorer-checkpoint", str(placeholder),
                                   "--device", "cpu", "--width", "4", "--height", "3",
                                   "--samples", "103", "--smooth-batch-size", "20",
                                   "--hessian-samples", "2", "--output", str(output)])
            self.assertEqual(status, 0)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["image_name"], "one.png")
            self.assertEqual(rows[0]["n_samples"], "103")
            self.assertEqual(rows[0]["ive_status"], "empirical_proxy")
            self.assertEqual(rows[0]["interval_status"], "finite")
            meta = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(meta["run_status"], "complete")
            self.assertEqual(meta["completed_images"], 1)
