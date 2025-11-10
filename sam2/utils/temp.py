import mrcfile
import torch
import numpy as np
import misc

# img_mean=(0.485, 0.456, 0.406)
# img_std=(0.229, 0.224, 0.225)
# image_size = 1024
# img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
# img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

# with mrcfile.open("/data/CryoETSeg/data/cdp/10001/256/tomogram/TS_0006.mrc", mode='r') as mrc:
#     images = mrc.data.astype(np.float32)

# images = torch.from_numpy(images).float()
# compute_device = torch.device("cuda")
# images = images.to(compute_device)
# img_mean = img_mean.to(compute_device)
# img_std = img_std.to(compute_device)
# video_height, video_width = images.shape[1], images.shape[2]

# if video_height > image_size or video_width > image_size:
#     print("Error: video_height or video_width is greater than image_size")
#     exit()

# images[images < 0] = 0
# images = (images / torch.max(images)) * 255.0

# # pad images to image_size x image_size
# images = torch.nn.functional.pad(images, (0, image_size - video_width, 0, image_size - video_height), mode='constant', value=0)
# images = images.unsqueeze(1)
# images = images.repeat(1, 3, 1, 1)
# images -= img_mean
# images /= img_std

# images, video_height, video_width = misc.load_video_frames_from_video_file(
#     video_path="/data/CryoETSeg/256_tomogram.avi",
#     image_size=1024,
#     offload_video_to_cpu=False,
#     img_mean=(0.485, 0.456, 0.406),
#     img_std=(0.229, 0.224, 0.225),
#     compute_device=torch.device("cuda"),
# )

images, video_height, video_width = misc.load_video_frames("/data/CryoETSeg/256_tomogram.avi", 1024, False)

print(images.shape)
print(video_height)
print(video_width)

images, video_height, video_width = misc.load_video_frames("/data/CryoETSeg/data/cdp/10001/256/tomogram/TS_0006.mrc", 1024, False)

print(images.shape)
print(video_height)
print(video_width)