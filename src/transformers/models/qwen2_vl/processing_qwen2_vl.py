# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""
Processor class for Qwen2-VL.
"""

from typing import List, Union, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from PIL import Image
import math

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, VideoInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging

# get fetch_audio from qwen_vl_utils's vision_process
from qwen_vl_utils import vision_process

logger = logging.get_logger(__name__)


class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class Qwen2VLProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2-VL processor which wraps a Qwen2-VL image processor and a Qwen2 tokenizer into a single processor.
    [`Qwen2VLProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2VLProcessor.__call__`] and [`~Qwen2VLProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "Qwen2VLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.audio_token = "<|audio_pad|>" if not hasattr(tokenizer, "audio_token") else tokenizer.audio_token # add the audio pad
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        audios: np.ndarray = None, # audio input from fetch_audio (changed from audio_array)
        audio_sample_rate: int = None, # sampling rate from fetch_audio (changed from sampling_rate)
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            audios (`np.ndarray`):
                The audio or batch of audio to be prepared. This should be raw audio numpy array that will be converted to log mel spectrogram.
            audio_sample_rates (`int`):
                The sample rate of the audio data.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **pixel_values_audios** -- Pixel values of audio spectrograms to be fed to a model. Returned when `audios` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
            - **audio_grid_thw** -- List of audio 3D grid in LLM. Returned when `audios` is not `None`.
        """
        # no you cannot do it like this. need to use the whisper as the encoder, don't increase the batch size but keep it as a list
        # all you need to do here is to estimate the token count
        # this can be done by going through the whisper tokenizer and determining the additional token count
        # need to handle the additional log mel you are creating
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if audios is not None:
            # calculate the length of the audio
            audio_length = audios.shape[0] / audio_sample_rate
            # number_encoder_tokens = 1500 * int(math.ceil(audio_length / 30.0)) # each 30 seconds of audio will create a new sample
            number_mel_frames = np.floor(audio_length * 100)
            if number_mel_frames % 2 != 0:
                number_mel_frames += 1
            number_encoder_tokens = int(number_mel_frames // 2)
            audio_values = torch.as_tensor(audios, dtype=torch.float32)
            audio_inputs = {
                "audio_values": audio_values,
                "audio_splits": torch.tensor([len(audios)], dtype=torch.int32),
                "number_encoder_tokens": torch.tensor([number_encoder_tokens], dtype=torch.int32),
                "audio_length": audios.shape[0] / audio_sample_rate
            }
        else:
            audio_inputs = {}


        # if audios is not None:
        #     audio_length = audios.shape[0]
        #     log_mel_spec = _convert_np_to_log_mel_spectrogram(SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_FRAMES, audios, DEVICE)
        #     # Convert log mel spectrogram to PIL images
        #     pil_images = convert_log_mel_to_pil_images(log_mel_spec)
        #     audio_processed = self.image_processor(images=pil_images, videos=None, **output_kwargs["images_kwargs"])
        #     # Rename keys to be audio-specific to avoid conflicts with image/video keys
        #     audio_inputs = {
        #         "pixel_values_audios": audio_processed["pixel_values"],
        #         "audio_grid_thw": audio_processed["image_grid_thw"]
        #     }
        #     audio_grid_thw = audio_processed["image_grid_thw"]
        # else:
        #     audio_inputs = {}
        #     audio_grid_thw = None
        #     audio_length = 0

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token, "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        if number_encoder_tokens:
            for i in range(len(text)):
                while self.audio_token in text[i]:
                    text[i] = text[i].replace(
                        self.audio_token, "<|placeholder|>" * number_encoder_tokens, 1
                    )
                text[i] = text[i].replace("<|placeholder|>", self.audio_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs, **audio_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

# def convert_log_mel_to_pil_images(log_mel_spec: torch.Tensor) -> List[Image.Image]:
#     """
#     Convert log mel spectrogram tensor to PIL Images.
    
#     Args:
#         log_mel_spec (torch.Tensor): Log mel spectrogram tensor of shape (B, n_mels, n_frames)
#                                     with values in range [-1, 1]
    
#     Returns:
#         List[Image.Image]: List of PIL Images, one for each batch item
#     """
#     # Move to CPU and convert to numpy
#     log_mel_np = log_mel_spec.cpu().numpy()
    
#     # Convert from [-1, 1] to [0, 255] for PIL Image
#     # First normalize to [0, 1], then scale to [0, 255]
#     log_mel_normalized = (log_mel_np + 1.0) / 2.0  # [-1, 1] -> [0, 1]
#     log_mel_uint8 = (log_mel_normalized * 255).astype(np.uint8)  # [0, 1] -> [0, 255]
    
#     pil_images = []
#     for i in range(log_mel_uint8.shape[0]):
#         # Create PIL Image from spectrogram (n_mels, n_frames)
#         # PIL expects (width, height) but we have (height, width), so we need to transpose
#         spec_image = log_mel_uint8[i]  # Shape: (n_mels, n_frames)
        
#         # Create PIL Image in 'L' (grayscale) mode
#         # PIL expects (width, height), so we pass (n_frames, n_mels)
#         pil_img = Image.fromarray(spec_image, mode='L')
#         pil_images.append(pil_img)
    
#     return pil_images

# def pad_and_split(
#     sr: int,
#     audio_np: np.ndarray,
#     device: Optional[Union[str, torch.device]] = 'cpu',
#     chunk_length_s: int = 30,
# ) -> torch.Tensor:
#     """
#     Split 1D audio array into fixed-length chunks of target_length where target_length = sr * chunk_length_s.
#     Right-pad the last chunk with zeros. Returns (B, T) float32 on device.
#     """

#     # samples within chunk (30 sec * 16kHz = 480000 samples)
#     target_length = int(sr * chunk_length_s)
#     # convert to tensor and send to device
#     x = torch.from_numpy(audio_np).to(device=device, dtype=torch.float32)
#     # calculate the chunk count
#     # do ceiling division to include partial chunk
#     # 5.1M samples with target_length = 500k -> 11 chunks
#     chunks = (x.shape[0] + target_length - 1) // target_length
#     padding = chunks * target_length - x.shape[0]
#     if padding:
#         x = F.pad(x,(0,padding))

#     return x.view(chunks, target_length)

# def _convert_np_to_log_mel_spectrogram(sr: int,
#                                       n_fft: int,
#                                       hop_length: int,
#                                       n_mels: int,
#                                       n_frames: int,
#                                       audio_np: np.ndarray,
#                                       device: Optional[Union[str, torch.device]]):
#     """
#     Convert a 1D audio numpy array to a log mel spectrogram.

#     Steps:
#     1. Check the audio size
#         1.1. If < 30 seconds, pad with zeros
#         1.2. If > 30 seconds, truncate to 30 seconds and create another sample with the remaining audio
#     2. Compute the mel spectrogram using librosa (FFT)
#     3. Convert to log scale (dB)
#     4. Normalize to [-1, 1]
#     5. Ensure the spectrogram has the correct shape (80,3000)
#     """

#     # Step 1: Check audio size, create new samples if needed
#     audios = pad_and_split(sr, audio_np, device).to(device) # (B,T)

#     # Step 2: Compute the log mel for each chunk

#     # window is to taper off the edges of each chunk to have smoother transitions across chunks
#     # torch.hann_window(8) returns tensor([0.0000, 0.1464, 0.5000, 0.8536, 1.0000, 0.8536, 0.5000, 0.1464])
#     # like a bell curve, high in middle, low at the edges
#     window = torch.hann_window(n_fft).to(device)
#     # window.shape torch.Size([400])

#     # transform with Fourier (B,T) -> (B,F+1,T+1)

#     stft = torch.stft(audios, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
#     # stft.shape torch.Size([2, 201, 3001])

#     # batch size of 2

#     # 201 freq bins → number of unique frequency bins along the y-axis
#     # Computed as N_FFT // 2 + 1
#     # Why +1? Because:
#     # - An N_FFT-point FFT produces N_FFT bins (0 … N_FFT-1), covering both + and – frequencies
#     # - For real audio, the spectrum is Hermitian: negative frequencies are redundant
#     # - So we keep only the non-redundant half: from 0 Hz (DC) up to Nyquist (sr/2)
#     #   That gives N_FFT//2 + 1 bins
#     # Example: N_FFT = 400 → 400//2 + 1 = 201 bins

#     # 3001 time frames -> count of frames on the time axis (x) calculated as T // HOP_LENGTH + 1
#     # Why +1? Because: stft is center = True by default meaning
#     # if stride is 12, seq len is 100 and window size is 25
#     # first window is centered at 0, so it covers -12 to +12
#     # last window is centered at 96, so it covers 84 to 100
#     # so we have 9 windows centered at 0,12,24,36,48,60,72,84,96
#     # formula: T // HOP_LENGTH + 1 which makes 480000 // 160 + 1 = 3001

#     magnitude = (stft.abs() ** 2)[..., :-1]
#     # magnitude.shape torch.Size([2, 201, 3000])
#     # we remove the last frame to make it 3000 frames

#     mel_np = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr/2.0).astype(np.float32)
#     mel = torch.tensor(mel_np, dtype=magnitude.dtype, device=device)
#     # mel.shape torch.Size([80, 201])
#     # mel represents the energy stored in that frequency bin for a specific time frame

#     mel_spec = torch.einsum('mf,bft->bmt', mel, magnitude)
#     # with batched, M,F @ B,F,T -> B,M,T
#     # mel_spec.shape torch.Size([2, 80, 3000])

#     log_mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
#     max_ele = log_mel_spec.amax(dim=(-2, -1), keepdim=True) # to get the max element in each spectrogram

#     # since the human ear can only capture 80 dB range, we clip the log mel spectrogram to max-8 as lower bound
#     # 8 and not 80 because we are in log10 scale
#     # this makes the log mel spectrogram values to be in the range of [max-8, max] where max ~ 0
#     log_mel_spec = torch.maximum(log_mel_spec, max_ele - 8.0) 
#     # then we normalize to [-1, 1] by dividing by 4
#     log_mel_spec = (log_mel_spec + 4.0) / 4.0

#     # Step 5: Ensure the spectrogram has the correct shape (80,3000) and not (80,3001)
#     # if < 3000 frames, pad with zeros
#     # if > 3000 frames, truncate to 3000 frames
#     T = log_mel_spec.shape[-1]
#     if T < n_frames:
#         log_mel_spec = F.pad(log_mel_spec, (0, n_frames - T))
#     elif T > n_frames:
#         log_mel_spec = log_mel_spec[..., :n_frames]
    
#     return log_mel_spec