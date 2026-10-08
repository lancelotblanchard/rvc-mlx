"""
Transcription of RVC's `infer/modules/vc/pipeline.py` (`Pipeline.vc` / `Pipeline.pipeline`) in PyTorch, used as the
end-to-end reference for `rvc_mlx.pipeline`. Differences from upstream are limited to plumbing:

* models are passed in (torch `TorchHubert` / `Synthesizer` references) instead of being globals;
* f0 comes from an injected `f0_fn(audio_pad) -> f0 (Hz)` (RMVPE is verified on its own);
* the sampling noise is zeroed (`deterministic`), so outputs are comparable;
* the result is returned as float (upstream's last step converts to int16 with the same peak rule).
"""

import faiss
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(rms2, torch.tensor(rate - 1))).numpy()
    return data2


class RefPipeline:
    def __init__(self, tgt_sr, x_pad, x_query, x_center, x_max, f0_fn):
        self.sr = 16000
        self.window = 160
        self.x_pad = x_pad
        self.t_pad = self.sr * x_pad
        self.t_pad_tgt = tgt_sr * x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * x_query
        self.t_center = self.sr * x_center
        self.t_max = self.sr * x_max
        self.f0_fn = f0_fn

    def get_f0(self, x, p_len, f0_up_key):
        f0_min, f0_max = 50, 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0 = self.f0_fn(x)
        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak

    def vc(self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, index_rate, version, protect):
        feats = torch.from_numpy(audio0).float()
        feats = feats.view(1, -1)
        with torch.no_grad():
            logits = model.extract_features(source=feats, output_layer=9 if version == "v1" else 12)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if index is not None and big_npy is not None and index_rate != 0:
            npy = feats[0].cpu().numpy()
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            feats = torch.from_numpy(npy).unsqueeze(0) * index_rate + (1 - index_rate) * feats
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len]).long()
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            T = feats.shape[1]
            prior = torch.zeros(1, net_g.enc_p.out_channels, T)
            nsf = torch.zeros(1, T * net_g.dec.upp, 1) if hasp else None
            if hasp:
                out = net_g.infer(feats, p_len, pitch, pitchf, sid, prior_noise=prior, nsf_noise=nsf)
            else:
                out = net_g.infer(feats, p_len, None, None, sid, prior_noise=prior)
            audio1 = out[0][0, 0].data.cpu().float().numpy()
        return audio1

    def pipeline(self, model, net_g, sid, audio, f0_up_key, index, big_npy, index_rate, if_f0, tgt_sr, rms_mix_rate, version, protect):
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query : t + self.t_query]
                        == audio_sum[t - self.t_query : t + self.t_query].min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key)
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf).unsqueeze(0).float()
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                audio_opt.append(
                    self.vc(model, net_g, sid, audio_pad[s : t + self.t_pad2 + self.window],
                            pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                            pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                            index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(model, net_g, sid, audio_pad[s : t + self.t_pad2 + self.window], None, None,
                            index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0 == 1:
            audio_opt.append(
                self.vc(model, net_g, sid, audio_pad[t:],
                        pitch[:, t // self.window :] if t is not None else pitch,
                        pitchf[:, t // self.window :] if t is not None else pitchf,
                        index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(model, net_g, sid, audio_pad[t:], None, None, index, big_npy, index_rate, version, protect)[
                    self.t_pad_tgt : -self.t_pad_tgt
                ]
            )
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt = audio_opt / audio_max
        return audio_opt


def build_faiss_index(vectors: np.ndarray, path: str) -> None:
    """Train an IVF index the way RVC's `train_index` does (n_ivf scaled down for small banks)."""
    n_ivf = max(1, min(int(16 * np.sqrt(vectors.shape[0])), vectors.shape[0] // 39))
    index = faiss.index_factory(vectors.shape[1], f"IVF{n_ivf},Flat")
    faiss.extract_index_ivf(index).nprobe = 1
    index.train(vectors)
    index.add(vectors)
    faiss.write_index(index, path)
