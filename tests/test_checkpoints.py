"""Checkpoint loading: meta-device construction and memory-mapped weights."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import sam2.build_sam as build_sam


class BufferedLinear(torch.nn.Linear):
    """A module with state that is not saved in its checkpoint."""

    def __init__(self):
        super().__init__(8, 8)
        self.register_buffer("scratch", torch.ones(2), persistent=False)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "weights.pt"

    def build(self, target, mode="eval", **arguments):
        config = OmegaConf.create({"_target_": target, **arguments})
        with patch.object(build_sam, "instantiate", wraps=build_sam.instantiate) as instantiate:
            model = build_sam._instantiate_with_checkpoint(config, self.path, mode)
        return model, instantiate.call_count

    def test_inference_model_is_built_once_on_the_meta_device(self):
        source = torch.nn.Linear(8, 8)
        torch.save({"model": source.state_dict()}, self.path)
        model, builds = self.build("torch.nn.Linear", in_features=8, out_features=8)
        self.assertEqual(builds, 1)
        self.assertTrue(torch.equal(model.weight, source.weight))
        self.assertTrue(torch.equal(model.bias, source.bias))

    def test_state_missing_from_the_checkpoint_falls_back_to_a_regular_build(self):
        torch.save({"model": BufferedLinear().state_dict()}, self.path)
        model, builds = self.build(__name__ + ".BufferedLinear")
        self.assertEqual(builds, 2)
        self.assertTrue(torch.equal(model.scratch, torch.ones(2)))

    def test_training_keeps_its_parameters(self):
        torch.save({"model": torch.nn.Linear(8, 8).state_dict()}, self.path)
        model = torch.nn.Linear(8, 8)
        weight = model.weight
        build_sam._load_checkpoint(model, self.path)
        self.assertIs(model.weight, weight)

    def test_mapped_weights_do_not_write_back_to_the_checkpoint(self):
        torch.save({"model": torch.nn.Linear(8, 8).state_dict()}, self.path)
        before = hashlib.sha256(self.path.read_bytes()).digest()
        model, _ = self.build("torch.nn.Linear", in_features=8, out_features=8)
        with torch.no_grad():
            model.weight.add_(1)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).digest(), before)

    def test_legacy_checkpoints_are_read_without_mapping(self):
        source = torch.nn.Linear(8, 8)
        torch.save({"model": source.state_dict()}, self.path, _use_new_zipfile_serialization=False)
        model, _ = self.build("torch.nn.Linear", in_features=8, out_features=8)
        self.assertTrue(torch.equal(model.weight, source.weight))


if __name__ == "__main__":
    unittest.main()
