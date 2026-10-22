import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.misc import PrefetchFrameLoader, TomogramFrameLoader

CHECKPOINT_PATH = "checkpoints/etsam_stage1_v1.pt"
CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_b+.yaml"

def load_video_frames_from_tomogram_data(
    tomogram_data,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda"),
):
    """Load the video frames from a mrc file."""

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    video_height, video_width = tomogram_data.shape[1], tomogram_data.shape[2]

    resized_images = []
    for i, image in enumerate(tomogram_data):
        image = Image.fromarray(image)
        image = image.resize((image_size, image_size))
        resized_images.append(np.array(image))

    tomogram_data = np.array(resized_images)

    tomogram_data = torch.from_numpy(tomogram_data).float()

    if video_height > image_size or video_width > image_size:
        print("Error: video_height or video_width is greater than image_size")
        exit()

    if not offload_video_to_cpu:
        tomogram_data = tomogram_data.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)

    tomogram_data[tomogram_data < 0] = 0
    tomogram_data = tomogram_data / torch.max(tomogram_data) #* 255.0

    tomogram_data = tomogram_data.unsqueeze(1)
    tomogram_data = tomogram_data.repeat(1, 3, 1, 1)
    tomogram_data -= img_mean
    tomogram_data /= img_std

    print(tomogram_data.shape)
    print(tomogram_data.dtype)

    return tomogram_data, video_height, video_width


def _write_slice(result, index, logits):
    if isinstance(result, np.ndarray):
        # Copy straight from the model into the result, without a CPU array in between.
        torch.from_numpy(result[index]).copy_(logits)
    else:
        result[index] = logits.cpu().numpy()


def _forget_expired_frames(predictor, inference_state, frame_idx):
    """Drop tracking outputs that forward propagation will not read again.

    Each frame attends to the memories of the previous num_maskmem frames (times
    the temporal stride) and to the object pointers of the previous
    max_obj_ptrs_in_encoder frames. Prompted (conditioning) frames are kept.
    """
    history = max(
        predictor.num_maskmem * predictor.memory_temporal_stride_for_eval,
        predictor.max_obj_ptrs_in_encoder,
    )
    for outputs in inference_state["output_dict_per_obj"].values():
        outputs["non_cond_frame_outputs"].pop(frame_idx - history, None)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(
    predictor,
    video_data,
    etsam1_seg_mask=None,
    convert_to_image=False,
    repeat_channels=True,
    prompt_method="grid",
    offload_video_to_cpu=True,
    offload_state_to_cpu=True,
    grid_interval=50,
    progress=None,
    output=None,
    mask_threshold=None,
):
    print("==> Preprocessing the input map")

    inference_state = predictor.init_state(
        video_path=video_data,
        async_loading_frames=False,
        convert_to_image=convert_to_image,
        repeat_channels=repeat_channels,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    frames = inference_state["images"]
    if inference_state["device"].type == "cuda" and isinstance(frames, TomogramFrameLoader):
        # Prepare the next frame on the CPU while the GPU tracks the current one.
        frames = inference_state["images"] = PrefetchFrameLoader(frames)
    try:
        return _prompt_and_propagate(
            predictor, inference_state, etsam1_seg_mask, prompt_method, grid_interval, progress,
            output, mask_threshold,
        )
    finally:
        if isinstance(frames, PrefetchFrameLoader):
            frames.close()


def _prompt_and_propagate(
    predictor, inference_state, etsam1_seg_mask, prompt_method, grid_interval, progress,
    output, mask_threshold,
):
    # print(inference_state["images"].shape)
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    num_frames = inference_state["num_frames"]
    grid_point_size = 1
    # pred_masks_on_prompt_frames = {}

    if prompt_method == "grid":
        input_frame_inds = np.arange(0, num_frames, grid_interval)  # Indices of frames to use as input masks
        print(f"using grid of points at every {grid_interval}th slice as mask prompt")
        object_ids_set = None
        for input_frame_idx in input_frame_inds:
            try:
                grid_points_Y = np.arange(0, height, grid_interval)
                grid_points_X = np.arange(0, width, grid_interval)
                grid_mask = np.zeros((height, width), dtype=np.uint8)
                for y in grid_points_Y:
                    for x in grid_points_X:
                        if y < height and x < width:
                            grid_mask[y:y+grid_point_size, x:x+grid_point_size] = 1 # Set a square mask at each grid point


                per_obj_input_mask = {
                    # 1: np.zeros((height, width), dtype=np.uint8)
                    # 1: np.random.randint(0, 2, (height, width)).astype(np.uint8)  # Random mask for testing
                    # grid points of mask
                    1: grid_mask  # Use the grid mask for testing

                }
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"In, failed to load input mask for frame {input_frame_idx=}. "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                ) from e
            # get the list of object ids to track from the first input frame
            if object_ids_set is None:
                object_ids_set = set(per_obj_input_mask)
            for object_id, object_mask in per_obj_input_mask.items():
                # check and make sure no new object ids appear only in later frames
                if object_id not in object_ids_set:
                    raise RuntimeError(
                        f"In , got a new {object_id=} appearing only in a "
                        f"later {input_frame_idx=} (but not appearing in the first frame). "
                        "Please add the `--track_object_appearing_later_in_video` flag "
                        "for VOS datasets that don't have all objects to track appearing "
                        "in the first frame (such as LVOS or YouTube-VOS)."
                    )
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    mask=object_mask,
                )
                # pred_masks_on_prompt_frames[input_frame_idx] = video_res_masks
    elif prompt_method == "grid_zero":
        input_frame_inds = [0]
        print(f"using grid of points at first slice as mask prompt (grid_interval={grid_interval})")
        object_ids_set = None
        for input_frame_idx in input_frame_inds:
            try:
                grid_points_Y = np.arange(0, height, grid_interval)
                grid_points_X = np.arange(0, width, grid_interval)
                grid_mask = np.zeros((height, width), dtype=np.uint8)
                for y in grid_points_Y:
                    for x in grid_points_X:
                        if y < height and x < width:
                            grid_mask[y:y+grid_point_size, x:x+grid_point_size] = 1 # Set a square mask at each grid point


                per_obj_input_mask = {
                    1: grid_mask
                }
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"In, failed to load input mask for frame {input_frame_idx=}. "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                ) from e
            # get the list of object ids to track from the first input frame
            if object_ids_set is None:
                object_ids_set = set(per_obj_input_mask)
            for object_id, object_mask in per_obj_input_mask.items():
                # check and make sure no new object ids appear only in later frames
                if object_id not in object_ids_set:
                    raise RuntimeError(
                        f"In , got a new {object_id=} appearing only in a "
                        f"later {input_frame_idx=} (but not appearing in the first frame). "
                        "Please add the `--track_object_appearing_later_in_video` flag "
                        "for VOS datasets that don't have all objects to track appearing "
                        "in the first frame (such as LVOS or YouTube-VOS)."
                    )
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    mask=object_mask,
                )
    elif prompt_method == "zero":
        input_frame_inds = [0]

        print("using zero initialized empty mask as prompt")
        object_ids_set = None
        for input_frame_idx in input_frame_inds:
            try:
                per_obj_input_mask = {
                    1: np.zeros((height, width), dtype=np.uint8)
                }
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"In, failed to load input mask for frame {input_frame_idx=}. "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                ) from e
            # get the list of object ids to track from the first input frame
            if object_ids_set is None:
                object_ids_set = set(per_obj_input_mask)
            for object_id, object_mask in per_obj_input_mask.items():
                # check and make sure no new object ids appear only in later frames
                if object_id not in object_ids_set:
                    raise RuntimeError(
                        f"In , got a new {object_id=} appearing only in a "
                        f"later {input_frame_idx=} (but not appearing in the first frame). "
                        "Please add the `--track_object_appearing_later_in_video` flag "
                        "for VOS datasets that don't have all objects to track appearing "
                        "in the first frame (such as LVOS or YouTube-VOS)."
                    )
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    mask=object_mask,
                )
    elif prompt_method == "etsam_stage1_partial":
        input_frame_inds = np.arange(0, num_frames, grid_interval)  # Indices of frames to use as input masks
        print(f"using ETSAM stage 1 predicted mask at every {grid_interval}th slice as prompt")
        object_ids_set = None
        for input_frame_idx in input_frame_inds:
            try:
                per_obj_input_mask = {
                    1: etsam1_seg_mask[input_frame_idx]
                }
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"In, failed to load input mask for frame {input_frame_idx=}. "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                ) from e
            # get the list of object ids to track from the first input frame
            if object_ids_set is None:
                object_ids_set = set(per_obj_input_mask)
            for object_id, object_mask in per_obj_input_mask.items():
                # check and make sure no new object ids appear only in later frames
                if object_id not in object_ids_set:
                    raise RuntimeError(
                        f"In , got a new {object_id=} appearing only in a "
                        f"later {input_frame_idx=} (but not appearing in the first frame). "
                        "Please add the `--track_object_appearing_later_in_video` flag "
                        "for VOS datasets that don't have all objects to track appearing "
                        "in the first frame (such as LVOS or YouTube-VOS)."
                    )
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    mask=object_mask,
                )
    else:
        print("Error: Invalid prompt method")
        exit()

    # run propagation throughout the video and write each slice into the result
    print("==> Running ETSAM inference on preprocessed map")
    if output is not None and len(object_ids_set) != 1:
        raise ValueError("output can only hold the result of a single object")
    dtype = np.float32 if mask_threshold is None else np.uint8
    per_obj_output_mask_full = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        for i, out_obj_id in enumerate(out_obj_ids):
            if out_obj_id not in per_obj_output_mask_full:
                per_obj_output_mask_full[out_obj_id] = (
                    output if output is not None else np.empty((num_frames, height, width), dtype=dtype)
                )
            logits = out_mask_logits[i, 0]
            if mask_threshold is not None:
                # Threshold on the GPU, then copy one byte per voxel instead of four.
                logits = logits > mask_threshold
            _write_slice(per_obj_output_mask_full[out_obj_id], out_frame_idx, logits)
        _forget_expired_frames(predictor, inference_state, out_frame_idx)
        if progress is not None:
            progress(out_frame_idx + 1, num_frames)

    # free up memory
    del predictor
    del inference_state
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    return per_obj_output_mask_full


def etsam_inference(
    tomogram_data,
    etsam1_seg_mask=None,
    config_path=CONFIG_PATH,
    ckpt_path=CHECKPOINT_PATH,
    convert_to_image=False,
    repeat_channels=True,
    prompt_method="grid_zero",
    offload_video_to_cpu=True,
    offload_state_to_cpu=True,
    grid_interval=50,
    device="cuda",
    progress=None,
    output=None,
    mask_threshold=None,
):
    """Run ETSAM inference on a single tomogram.

    progress(slices_done, total_slices) is called after each slice.
    output optionally receives the (num_frames, H, W) logits, one slice at a time:
    an array, or any object that accepts output[index] = slice.
    With mask_threshold, the result is a uint8 mask (logits > mask_threshold).
    """
    hydra_overrides_extra = [
        "++model.non_overlap_masks=false"
    ]

    print("==> Loading ETSAM with checkpoint: ", ckpt_path)

    predictor = build_sam2_video_predictor(
        config_file=config_path,
        ckpt_path=ckpt_path,
        apply_postprocessing=False,
        hydra_overrides_extra=hydra_overrides_extra,
        vos_optimized=False,
        device=device,
    )

    return vos_inference(
        predictor,
        tomogram_data,
        etsam1_seg_mask,
        convert_to_image=convert_to_image,
        repeat_channels=repeat_channels,
        prompt_method=prompt_method,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        grid_interval=grid_interval,
        progress=progress,
        output=output,
        mask_threshold=mask_threshold,
    )
