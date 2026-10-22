"""The two-stage pipeline must match the original whole-volume arithmetic byte for byte.

Run all tests with: python -m unittest discover -s tests
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

import etsam
from sam2.utils.volume import NormalizedTomogram


def original_min_max_positive_values(tomogram):
    return etsam.min_max_postive_values_normalize(np.array(tomogram, dtype=np.float32))


def original_fusion(tomogram, logits):
    clipped = logits.copy()
    clipped[clipped < -5] = -5
    clipped[clipped > 5] = 5
    return tomogram + etsam.min_max_normalize(clipped)


def stack(volume):
    return np.stack([volume[index] for index in range(len(volume))])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(732)

    def assertBytesEqual(self, actual, expected):
        self.assertEqual((actual.dtype, actual.shape), (expected.dtype, expected.shape))
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_normalization_by_slice(self):
        for dtype in (np.float32, np.float64, np.int16, np.int8, np.dtype(">f4")):
            tomogram = (self.rng.standard_normal((9, 19, 23)) * 40).astype(dtype)
            before = tomogram.tobytes()
            expected = original_min_max_positive_values(tomogram)
            normalized = NormalizedTomogram(tomogram)
            self.assertBytesEqual(stack(normalized), expected)
            self.assertBytesEqual(stack(normalized[:, 2:14, 3:17]), expected[:, 2:14, 3:17])
            self.assertEqual(tomogram.tobytes(), before)

    def test_softplus_in_place(self):
        # Large enough to be split across threads and vectorized.
        tomogram = self.rng.standard_normal((40, 64, 64)).astype(np.float32) * 30
        softplus = torch.nn.functional.softplus(torch.from_numpy(tomogram)).numpy()
        expected = (softplus - np.min(softplus)) / (np.max(softplus) - np.min(softplus))
        self.assertBytesEqual(etsam.softplus_minmax_normalize(tomogram.copy()), expected)

    def run_pipeline(self, tomogram, logits, stage2_prompt):
        """Run both stages with a stand-in model: stage 1 returns logits,
        stage 2 returns its input times two."""
        calls = []

        def etsam_inference(volume, prompts=None, *, output, prompt_method, **kwargs):
            calls.append((stack(volume), prompts))
            for index in range(len(volume)):
                output[index] = logits[index] if len(calls) == 1 else volume[index] * 2
            return {1: output}

        with patch.object(etsam.inference, "etsam_inference", side_effect=etsam_inference):
            result = etsam.run_two_stage_inference_on_tomogram(
                tomogram, "config", "stage1", "stage2", "grid_zero", stage2_prompt, 5, -0.25
            )
        return result, calls[1]

    def test_fusion_and_prompts(self):
        tomogram = self.rng.random((13, 19, 23), dtype=np.float32)
        logits = self.rng.standard_normal(tomogram.shape).astype(np.float32) * 7
        fused = original_fusion(tomogram, logits)
        for stage2_prompt in ("zero", "etsam_stage1_partial"):
            result, (stage2_input, prompts) = self.run_pipeline(tomogram, logits, stage2_prompt)
            self.assertBytesEqual(stage2_input, fused)
            self.assertBytesEqual(result, fused * 2)
            if stage2_prompt == "etsam_stage1_partial":
                self.assertEqual(set(prompts), {0, 5, 10})
                for index, prompt in prompts.items():
                    self.assertBytesEqual(prompt, (logits[index] > -0.25).astype(np.uint8))

    def test_split_processing_writes_each_quadrant_in_place(self):
        tomogram = self.rng.standard_normal((7, 19, 23)).astype(np.float32)

        def two_stages(volume, *args, output, **kwargs):
            output[...] = stack(volume)

        with patch.object(etsam, "ensure_checkpoints", return_value=("a", "b")), patch.object(
            etsam, "run_two_stage_inference_on_tomogram", side_effect=two_stages
        ):
            result = etsam.segment(tomogram, split_processing=True)
        self.assertBytesEqual(result, original_min_max_positive_values(tomogram))


if __name__ == "__main__":
    unittest.main()
