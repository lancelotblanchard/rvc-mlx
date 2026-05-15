# from typing import Optional
# import librosa
# import mlx.core as mx
# import mlx.nn as nn
#
# from rvc_mlx.stft import stft
# from rvc_mlx.windows import hann
#
#
# class MelSpectrogram(nn.Module):
#     def __init__(
#         self,
#         is_half: bool,
#         n_mel_channels: bool,
#         sampling_rate: int,
#         win_length: int,
#         hop_length: int,
#         n_fft: Optional[int] = None,
#         mel_fmin: int = 0,
#         mel_fmax: Optional[int] = None,
#         clamp: float = 1e-5,
#     ):
#         super().__init__()
#         n_fft = win_length if n_fft is None else n_fft
#         self.hann_window = {}
#         mel_basis = librosa.filters.mel(
#             sr=sampling_rate,
#             n_fft=n_fft,
#             n_mels=n_mel_channels,
#             fmin=mel_fmin,
#             fmax=mel_fmax,
#             htk=True,
#         )
#         mel_basis = mx.array(mel_basis)
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.sampling_rate = sampling_rate
#         self.n_mel_channels = n_mel_channels
#         self.clamp = clamp
#         self.is_half = is_half
#
#     def __call__(
#         self,
#         audio: mx.array,
#         keyshift: int = 0,
#         speed: int = 1,
#         center: bool = True,
#     ):
#         factor = 2 ** (keyshift / 12)
#         n_fft_new = int(round(self.n_fft * factor))
#         win_length_new = int(round(self.win_length * factor))
#         hop_length_new = int(round(self.hop_length * speed))
#         keyshift_key = str(keyshift)
#         if keyshift_key not in self.hann_window:
#             self.hann_window[keyshift_key] = hann(win_length_new)
#
#         fft = stft(
#             audio,
#             n_fft=n_fft_new,
#             hop_length=hop_length_new,
#             win_length=win_length_new,
#             window=self.han_window[keyshift_key],
#             center=center,
#         )
#         magnitude = mx.sqrt(fft.real**2 + fft.imag**2)
#         if keyshift != 0:
#             size = self.n_fft // 2 + 1
#             resize = magnitude.shape[1]
#             if resize < size:
#                 magnitude = pad...
#
#
# class TestTest:
#     def forward(self, audio, keyshift=0, speed=1, center=True):
#         factor = 2 ** (keyshift / 12)
#         n_fft_new = int(np.round(self.n_fft * factor))
#         win_length_new = int(np.round(self.win_length * factor))
#         hop_length_new = int(np.round(self.hop_length * speed))
#         keyshift_key = str(keyshift) + "_" + str(audio.device)
#         if keyshift_key not in self.hann_window:
#             self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)
#
#         fft = torch.stft(
#             audio,
#             n_fft=n_fft_new,
#             hop_length=hop_length_new,
#             win_length=win_length_new,
#             window=self.hann_window[keyshift_key],
#             center=center,
#             return_complex=True,
#         )
#         magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
#         if keyshift != 0:
#             size = self.n_fft // 2 + 1
#             resize = magnitude.size(1)
#             if resize < size:
#                 magnitude = torch.nn.functional.pad(magnitude, (0, 0, 0, size - resize))
#             magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
#         mel_output = torch.matmul(self.mel_basis, magnitude)
#         if self.is_half == True:
#             mel_output = mel_output.half()
#         log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
#         return log_mel_spec
