"""Propagation loop of inference.vos_inference, run with a stand-in predictor."""

import unittest

import numpy as np
import torch

import inference
from sam2.utils.misc import PrefetchFrameLoader, load_video_frames_from_numpy_array_as_float_numpy


class FakePredictor:
    """Returns each frame times two as its logits, like SAM2's predictor API."""

    num_maskmem = 7
    memory_temporal_stride_for_eval = 1
    max_obj_ptrs_in_encoder = 16

    def init_state(self, video_path, **kwargs):
        self.volume = video_path
        self.most_frames_kept = 0
        num_frames, height, width = video_path.shape
        return {
            "images": video_path,
            "num_frames": num_frames,
            "video_height": height,
            "video_width": width,
            "device": torch.device("cpu"),
            "output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
        }

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        inference_state["output_dict_per_obj"][0]["cond_frame_outputs"][frame_idx] = mask

    def propagate_in_video(self, inference_state):
        kept = inference_state["output_dict_per_obj"][0]["non_cond_frame_outputs"]
        for index in range(inference_state["num_frames"]):
            kept[index] = index
            self.most_frames_kept = max(self.most_frames_kept, len(kept))
            frame = np.asarray(self.volume[index], dtype=np.float32)
            yield index, [1], torch.from_numpy(frame * 2)[None, None]


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.volume = np.random.default_rng(5).standard_normal((40, 9, 11)).astype(np.float32)

    def test_slices_are_written_in_order(self):
        progress = []
        result = inference.vos_inference(
            FakePredictor(), self.volume, prompt_method="zero",
            progress=lambda done, total: progress.append((done, total)),
        )
        np.testing.assert_array_equal(result[1], self.volume * 2)
        self.assertEqual(progress, [(i + 1, 40) for i in range(40)])

    def test_masks_are_thresholded_like_numpy(self):
        for threshold in (-0.25, 0.1, 1e-45, float("inf")):
            result = inference.vos_inference(
                FakePredictor(), self.volume, prompt_method="zero", mask_threshold=threshold
            )
            expected = (self.volume * 2 > threshold).astype(np.uint8)
            self.assertEqual(result[1].tobytes(), expected.tobytes())

    def test_tracking_history_is_bounded(self):
        predictor = FakePredictor()
        inference.vos_inference(predictor, self.volume, prompt_method="zero")
        self.assertEqual(predictor.most_frames_kept, 17)


class FrameLoaderTests(unittest.TestCase):
    def test_frames_prepared_on_demand_match_the_whole_video(self):
        rng = np.random.default_rng(3)
        for shape, repeat_channels in (((6, 17, 21), True), ((6, 17, 21, 3), False)):
            volume = rng.standard_normal(shape).astype(np.float32)
            for size in (16, 32):
                load = lambda offload: load_video_frames_from_numpy_array_as_float_numpy(
                    volume, size, offload, compute_device=torch.device("cpu"),
                    repeat_channels=repeat_channels,
                )
                video, _, _ = load(offload=False)
                frames, height, width = load(offload=True)
                self.assertEqual((len(frames), height, width), (6, 17, 21))
                for index in (5, 0, 3, 0):
                    self.assertTrue(torch.equal(frames[index], video[index]))

    def test_prefetching_returns_the_requested_frames(self):
        volume = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
        frames = PrefetchFrameLoader(volume)
        try:
            for index in (0, 3, 1, 2, 3, 3, 5, 0, 1):
                np.testing.assert_array_equal(frames[index], volume[index])
        finally:
            frames.close()


if __name__ == "__main__":
    unittest.main()
