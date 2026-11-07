"""Center-crop + nearest-neighbour downscale transform shared by the video datasets and the pretraining preprocess task."""
import torch


def extract_add_time_info_for_video(video, kwargs):
    # Todo: get this information from dataset (-> conversion stage)

    # video is (F, C, H, W): frames is dim 0, not the channel dim.
    nr_frames = video.shape[0]
    if "attrs" in kwargs and "fps" in kwargs["attrs"]:
        fps = kwargs["attrs"]["fps"]
    else:
        video_duration = 1.0 # at roughly 60 bpm
        fps = nr_frames / video_duration
    video_relative_start = 0.0
    slice_pos_rel = 0.0
    return torch.tensor(
        [nr_frames, fps, video_relative_start, slice_pos_rel], device=video.device,
    ) # here we can also add relative timeing later if video doesn't start beginning of systole


class SpatialCenterCropAndDownscale:
    def __init__(self, size_before_downscale : tuple, size_after_downscale : tuple, normalization : str = "none",  **kwargs):

        if isinstance(size_after_downscale, int):
            size_after_downscale = (size_after_downscale, size_after_downscale)
        else:
            size_after_downscale = tuple(size_after_downscale)

        if isinstance(size_before_downscale, int):
            size_before_downscale = (size_before_downscale, size_before_downscale)
        else:
            size_before_downscale = tuple(size_before_downscale)

        self.size_before_downscale = size_before_downscale  # (height, width) 
        self.size_after_downscale = size_after_downscale  # (height, width)
        self.normalization = normalization

    def __call__(self, video, warped_noise_sample, **kwargs):
        f, c, h, w = video.shape # same shape for the warped_noise_sample

        # first we center crop to the target size
        target_h_shape, target_w_shape = self.size_before_downscale
        h1 = min(max(0, h//2 - target_h_shape//2), h-1)
        h2 = max(min(h-1, h//2 + target_h_shape//2), 0)
        w1 = min(max(0, w//2 - target_w_shape//2), w-1)
        w2 = max(min(w-1, w//2 + target_w_shape//2), 0)
        video_cropped = video[:, :, h1:h2, w1:w2]
        warped_noise_sample_cropped = warped_noise_sample[:, :, h1:h2, w1:w2]

        if self.normalization == "gaussian":
            # normalize to zero mean and unit variance
            video_cropped = (video_cropped - video_cropped.mean()) / (video_cropped.std() + 1e-8)
        elif self.normalization == "uniform_-1_1":
            # normalize to [-1, 1]
            video_cropped = 2 * (video_cropped - video_cropped.min()) / (video_cropped.max() - video_cropped.min() + 1e-8) - 1
        elif self.normalization == "uniform_0_1":
            # normalize to [0, 1]
            video_cropped = (video_cropped - video_cropped.min()) / (video_cropped.max() - video_cropped.min() + 1e-8)
        elif self.normalization == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization}")
        
        # now we apply downsampling using torch.nn interpolate with nearest neighbor
        if self.size_before_downscale != self.size_after_downscale:
            video_processed = torch.nn.functional.interpolate(video_cropped, size=self.size_after_downscale, mode='nearest')
            warped_noise_sample_processed = torch.nn.functional.interpolate(warped_noise_sample_cropped, size=self.size_after_downscale, mode='nearest')
        else:
            video_processed = video_cropped
            warped_noise_sample_processed = warped_noise_sample_cropped

        return {"video": video_processed, "warped_noise_sample": warped_noise_sample_processed, 
            "added_time_ids" : extract_add_time_info_for_video(video, kwargs),
            **kwargs}
