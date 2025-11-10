import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

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

    # print(inference_state["images"].shape)
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    num_frames = inference_state["num_frames"]
    grid_interval = 50
    grid_point_size = 1
    # pred_masks_on_prompt_frames = {}

    if prompt_method == "grid":
        input_frame_inds = np.arange(0, num_frames, grid_interval)  # Indices of frames to use as input masks
        print("using grid of points at every 50th slice as mask prompt")
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
        print("using grid of points at first slice as mask prompt")
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
        interval = 50
        input_frame_inds = np.arange(0, num_frames, interval)  # Indices of frames to use as input masks
        print("using ETSAM stage 1 predicted mask at every 50th slice as prompt")
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

    # run propagation throughout the video and collect the results in a dict
    print("==> Running ETSAM inference on preprocessed map")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        per_obj_output_mask = {
            out_obj_id: out_mask_logits[i].cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask

    per_obj_output_mask_full = {}
    for object_id in object_ids_set:
        per_obj_output_mask_full[object_id] = []
        for frame_idx in video_segments.keys():
            per_obj_output_mask_full[object_id].append(video_segments[frame_idx][object_id])
    
    for object_id in object_ids_set:
            per_obj_output_mask_full[object_id] = np.array(per_obj_output_mask_full[object_id]).squeeze(1)

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
):
    """Run ETSAM inference on a single tomogram."""
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
    )
