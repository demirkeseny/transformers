# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video processing utilities: loading, processing, batching and more."""

import io
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, NewType, Optional, Tuple, Union

import numpy as np

from .image_utils import ChannelDimension, get_image_size, is_valid_image
from .utils import (
    is_numpy_array,
    is_torch_available,
    is_torch_tensor,
    is_vision_available,
    logging,
)


if is_vision_available():
    import PIL.Image

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)

URL = NewType("URL", str)
Path = NewType("Path", str)

VideoInput = Union[
    List["PIL.Image.Image"],
    np.ndarray,
    "torch.Tensor",
    List[np.ndarray],
    List["torch.Tensor"],
    List[List["PIL.Image.Image"]],
    List[List[np.ndarray]],
    List[List["torch.Tensor"]],
    URL,
    List[URL],
    List[List[URL]],
    Path,
    List[Path],
    List[List[Path]],
]


@dataclass
class VideoMetadata(Mapping):
    """Video metadata containing frame information."""
    
    total_num_frames: Optional[int] = None
    fps: Optional[float] = None
    duration: Optional[float] = None
    frames_indices: Optional[List[int]] = None
    height: Optional[int] = None
    width: Optional[int] = None

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def __len__(self):
        return len(self.__dict__)


def is_valid_video_frame(frame):
    """Check if frame is a valid video frame."""
    return is_valid_image(frame)


def is_valid_video(video):
    """Check if video is valid."""
    if isinstance(video, (list, tuple)):
        return all(is_valid_video_frame(frame) for frame in video)
    elif is_numpy_array(video) or is_torch_tensor(video):
        return video.ndim == 4  # (T, H, W, C)
    return False


def valid_videos(videos):
    """Check if all videos are valid."""
    if isinstance(videos, (list, tuple)):
        return all(is_valid_video(video) for video in videos)
    return is_valid_video(videos)


def is_batched_video(videos):
    """Check if videos are batched."""
    if isinstance(videos, (list, tuple)):
        return is_valid_video(videos[0])
    elif (is_numpy_array(videos) or is_torch_tensor(videos)) and videos.ndim == 5:
        return True
    return False


def is_scaled_video(video: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].
    """
    # It's possible the video has pixel values in [0, 255] but is of floating type
    return np.min(video) >= 0 and np.max(video) <= 1


def convert_pil_frames_to_video(videos: List[VideoInput]) -> List[Union[np.ndarray, "torch.Tensor"]]:
    """
    Given a batch of videos, converts each video to a 4D array. If video is already in array type,
    it is simply returned. We assume that all inputs in the list are in the same format, based on the type of the first element.

    Args:
        videos (`VideoInput`):
            Video inputs to turn into a list of videos.
    """

    if not (isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0])):
        return videos

    video_converted = []
    for video in videos:
        video = [np.array(frame) for frame in video]
        video = np.stack(video)
        video_converted.append(video)
    return video_converted


def make_batched_videos(videos) -> List[Union[np.ndarray, "torch.Tensor", "URL", "Path"]]:
    """
    Ensure that the input is a list of videos. If the input is a single video, it is converted to a list of length 1.
    If the input is a batch of videos, it is converted to a list of 4D video arrays. Videos passed as list `PIL.Image`
    frames are converted to 4D arrays.

    We assume that all inputs in the list are in the same format, based on the type of the first element.

    Args:
        videos (`VideoInput`):
            Video inputs to turn into a list of videos.
    """
    # Early exit for deeply nested list of image frame paths. We shouldn't flatten them
    try:
        if isinstance(videos[0][0], list) and isinstance(videos[0][0][0], str):
            return [image_paths for sublist in videos for image_paths in sublist]
    except (IndexError, TypeError):
        pass

    if isinstance(videos, str) or is_valid_video(videos):
        return convert_pil_frames_to_video([videos])
    # only one frame passed, thus we unsqueeze time dim
    elif is_valid_image(videos):
        return [np.array(videos)[None, ...]]
    elif not isinstance(videos, list):
        raise ValueError(
            f"Invalid video input. Expected either a list of video frames or an input of 4 or 5 dimensions, but got"
            f" type {type(videos)}."
        )

    # Recursively flatten any nested structure
    flat_videos_list = []
    for item in videos:
        if isinstance(item, str) or is_valid_video(item):
            flat_videos_list.append(item)
        elif isinstance(item, list) and item:
            flat_videos_list.extend(make_batched_videos(item))

    flat_videos_list = convert_pil_frames_to_video(flat_videos_list)
    return flat_videos_list


def make_batched_metadata(videos: VideoInput, video_metadata: Union[VideoMetadata, dict]):
    """Create batched metadata for videos."""
    if video_metadata is None:
        # Create default metadata and fill attributes we can infer from given video
        video_metadata = [
            {
                "total_num_frames": len(video),
                "fps": None,
                "duration": None,
                "frames_indices": list(range(len(video))),
                "height": get_video_size(video)[0] if is_valid_video(video) else None,
                "width": get_video_size(video)[1] if is_valid_video(video) else None,
            }
            for video in videos
        ]

    if isinstance(video_metadata, list):
        # Flatten if nested list
        if isinstance(video_metadata[0], list):
            video_metadata = [
                VideoMetadata(**metadata) for metadata_list in video_metadata for metadata in metadata_list
            ]
        # Simply wrap in VideoMetadata if simple dict
        elif isinstance(video_metadata[0], dict):
            video_metadata = [VideoMetadata(**metadata) for metadata in video_metadata]
    else:
        # Create a batched list from single object
        video_metadata = [VideoMetadata(**video_metadata)]
    return video_metadata


def get_video_size(video: np.ndarray, channel_dim: Optional[ChannelDimension] = None) -> Tuple[int, int]:
    """
    Get the size (height, width) of a video.

    Args:
        video (`np.ndarray`):
            The video to get the size of. Expected to be 4D (T, H, W, C).
        channel_dim (`ChannelDimension`, *optional*):
            The channel dimension of the video. If not provided, it will be inferred.

    Returns:
        `Tuple[int, int]`: The height and width of the video.
    """
    if video.ndim == 4:
        # Assuming (T, H, W, C) format
        return video.shape[1], video.shape[2]
    elif video.ndim == 3:
        # Single frame (H, W, C)
        return video.shape[0], video.shape[1]
    else:
        raise ValueError(f"Unsupported video shape: {video.shape}")


def get_uniform_frame_indices(total_num_frames: int, num_frames: Optional[int] = None):
    """Get uniform frame indices for sampling."""
    if num_frames is None:
        return list(range(total_num_frames))
    
    if num_frames > total_num_frames:
        raise ValueError(f"Cannot sample {num_frames} frames from video with {total_num_frames} frames")
    
    indices = np.linspace(0, total_num_frames - 1, num_frames, dtype=int)
    return indices.tolist()


def group_videos_by_shape(videos, disable_grouping=False):
    """Group videos by their shape."""
    if disable_grouping:
        return {i: [video] for i, video in enumerate(videos)}, {i: (i, 0) for i in range(len(videos))}
    
    grouped_videos = {}
    grouped_videos_index = {}
    
    for i, video in enumerate(videos):
        shape = video.shape
        if shape not in grouped_videos:
            grouped_videos[shape] = []
        grouped_videos[shape].append(video)
        grouped_videos_index[i] = (shape, len(grouped_videos[shape]) - 1)
    
    return grouped_videos, grouped_videos_index


def reorder_videos(grouped_videos, grouped_videos_index):
    """Reorder videos based on their original indices."""
    reordered_videos = []
    for i in sorted(grouped_videos_index.keys()):
        shape, idx = grouped_videos_index[i]
        reordered_videos.append(grouped_videos[shape][idx])
    return reordered_videos


def load_video(
    video: VideoInput,
    num_frames: Optional[int] = None,
    fps: Optional[Union[int, float]] = None,
    backend: str = "pyav",
    sample_indices_fn: Optional[Callable] = None,
    **kwargs,
) -> np.array:
    """
    Loads `video` to a numpy array.

    Args:
        video (`VideoInput`):
            The video to convert to the numpy array format. Can be a link to video or local path.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. If not passed, the whole video is loaded.
        fps (`int` or `float`, *optional*):
            Number of frames to sample per second. Should be passed only when `num_frames=None`.
            If not specified and `num_frames==None`, all frames are sampled.
        backend (`str`, *optional*, defaults to `"pyav"`):
            Backend to use for loading video.
        sample_indices_fn (`Callable`, *optional*):
            Function to sample frame indices.

    Returns:
        `np.array`: The loaded video as a numpy array.
    """
    # Placeholder implementation - in practice this would use actual video loading backends
    raise NotImplementedError("Video loading functionality is not implemented in this minimal version")


def to_channel_dimension_format(
    video: np.ndarray,
    channel_dim: ChannelDimension,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Convert video to the specified channel dimension format.
    """
    # Simplified implementation - in practice this would handle channel dimension conversion
    return video
