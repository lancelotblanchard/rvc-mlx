"""
Tests for `Pipeline.get_f0`.

`get_f0` is pure NumPy post-processing on top of `RMVPE.infer_from_audio`, so there is no PyTorch counterpart
to bridge against. Instead we exercise the documented contract directly with concrete inputs:

  * pitch-shift via `f0_up_key` multiplies the raw f0 by `2 ** (k / 12)`,
  * the `inp_f0` argument replaces a slice of the (already shifted) f0 with linearly-interpolated user values,
  * the mel-coarse mapping clamps to [1, 255] and maps unvoiced (f0 == 0) to bin 1,
  * `f0bak` is the un-quantised post-shift / post-splice f0.

The RMVPE network itself is stubbed (`_StubRMVPE`) so each test pins the f0 the post-processing sees. The network
side is covered separately in `tests/test_rmvpe.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from rvc_mlx.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


class _StubRMVPE:
    """Stand-in for `RMVPE` whose `infer_from_audio` returns a pre-set 1D f0 array."""

    def __init__(self, f0: np.ndarray):
        self._f0 = f0

    def infer_from_audio(self, audio, thred=0.03):  # noqa: ARG002 — match the real signature
        return self._f0.astype(np.float64).copy()


def _make_pipeline(f0_raw: np.ndarray, x_pad: int = 1) -> Pipeline:
    """Build a Pipeline without running `__init__` (which would try to load a checkpoint)."""
    p = object.__new__(Pipeline)
    p.x_pad = x_pad
    p.x_query = 6
    p.x_center = 38
    p.x_max = 41
    p.is_half = False
    p.sr = 16_000
    p.window = 16
    p.t_pad = p.sr * p.x_pad
    p.t_pad_tgt = p.sr * p.x_pad
    p.t_pad2 = p.t_pad * 2
    p.t_query = p.sr * p.x_query
    p.t_center = p.sr * p.x_center
    p.t_max = p.sr * p.x_max
    p.model_rmvpe = _StubRMVPE(f0_raw)
    return p


# Constants derived from the get_f0 implementation; pinned here so the tests document the contract.
_F0_MEL_MIN = 1127 * np.log(1 + 50 / 700)
_F0_MEL_MAX = 1127 * np.log(1 + 1100 / 700)


def _expected_coarse(f0_shifted: np.ndarray) -> np.ndarray:
    """Reference cents-to-coarse mapping, written from the spec rather than copy-pasted from the implementation.

    Voiced bins span 1..255; unvoiced and below-range f0 collapse to bin 1; above-range clamps to 255.
    """
    mel = 1127 * np.log(1 + f0_shifted / 700)
    scaled = np.where(
        mel > 0,
        (mel - _F0_MEL_MIN) * 254 / (_F0_MEL_MAX - _F0_MEL_MIN) + 1,
        mel,
    )
    scaled = np.clip(scaled, 1, 255)
    return np.rint(scaled).astype(np.int32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetF0NoOp:
    """`f0_up_key=0`, no `inp_f0`: only mel-coarse quantisation runs."""

    def test_shapes_and_dtypes(self):
        f0_raw = np.array([0.0, 100.0, 200.0, 440.0, 880.0], dtype=np.float64)
        pipe = _make_pipeline(f0_raw)
        f0_coarse, f0bak = pipe.get_f0(x=np.zeros(16000), f0_up_key=0)

        assert f0_coarse.dtype == np.int32
        assert f0bak.dtype == np.float64
        assert f0_coarse.shape == f0_raw.shape
        assert f0bak.shape == f0_raw.shape

    def test_f0bak_is_raw_f0_when_unshifted(self):
        f0_raw = np.array([0.0, 100.0, 200.0, 440.0, 880.0])
        pipe = _make_pipeline(f0_raw)
        _, f0bak = pipe.get_f0(x=np.zeros(16000), f0_up_key=0)
        np.testing.assert_allclose(f0bak, f0_raw)

    def test_coarse_matches_spec(self):
        # A spread of voiced f0 + one unvoiced; values picked to exercise the mid-range mapping (no clamping).
        f0_raw = np.array([0.0, 80.0, 200.0, 500.0, 900.0])
        pipe = _make_pipeline(f0_raw)
        f0_coarse, _ = pipe.get_f0(x=np.zeros(16000), f0_up_key=0)
        np.testing.assert_array_equal(f0_coarse, _expected_coarse(f0_raw))

    def test_unvoiced_maps_to_one(self):
        f0_raw = np.zeros(10)
        pipe = _make_pipeline(f0_raw)
        f0_coarse, f0bak = pipe.get_f0(x=np.zeros(16000), f0_up_key=0)
        np.testing.assert_array_equal(f0_coarse, np.ones(10, dtype=np.int32))
        np.testing.assert_array_equal(f0bak, np.zeros(10))


class TestGetF0PitchShift:
    """`f0_up_key` shifts the raw f0 by `2 ** (k / 12)`. This is the part users hear as transposition."""

    def test_shift_up_by_octave_doubles_f0bak(self):
        f0_raw = np.array([100.0, 200.0, 0.0, 440.0])
        pipe = _make_pipeline(f0_raw)
        _, f0bak = pipe.get_f0(x=np.zeros(16000), f0_up_key=12)
        # Unvoiced (0) stays unvoiced after the multiplicative shift.
        np.testing.assert_allclose(f0bak, f0_raw * 2.0)

    def test_shift_down_by_octave_halves_f0bak(self):
        f0_raw = np.array([200.0, 400.0, 800.0])
        pipe = _make_pipeline(f0_raw)
        _, f0bak = pipe.get_f0(x=np.zeros(16000), f0_up_key=-12)
        np.testing.assert_allclose(f0bak, f0_raw / 2.0)

    def test_coarse_uses_shifted_not_raw(self):
        f0_raw = np.array([100.0, 200.0, 400.0])
        pipe = _make_pipeline(f0_raw)
        f0_coarse, _ = pipe.get_f0(x=np.zeros(16000), f0_up_key=7)  # perfect fifth
        np.testing.assert_array_equal(f0_coarse, _expected_coarse(f0_raw * (2 ** (7 / 12))))


class TestGetF0Clamping:
    """f0 outside [50, 1100] Hz is clamped: below -> coarse=1, above -> coarse=255."""

    def test_below_f0_min_clamps_to_one(self):
        # Voiced but below 50 Hz: mel > 0 but the (mel - f0_mel_min) term is negative, so scaled <= 1 -> clamp to 1.
        f0_raw = np.array([10.0, 30.0, 49.0])
        pipe = _make_pipeline(f0_raw)
        f0_coarse, _ = pipe.get_f0(x=np.zeros(16000), f0_up_key=0)
        np.testing.assert_array_equal(f0_coarse, np.ones(3, dtype=np.int32))

    def test_above_f0_max_clamps_to_255(self):
        f0_raw = np.array([1200.0, 2000.0, 8000.0])
        pipe = _make_pipeline(f0_raw)
        f0_coarse, _ = pipe.get_f0(x=np.zeros(16000), f0_up_key=0)
        np.testing.assert_array_equal(f0_coarse, np.full(3, 255, dtype=np.int32))


class TestGetF0InpF0Splice:
    """`inp_f0` overrides a slice of the (post-shift) f0 with user values, linearly interpolated frame-by-frame.

    Splice geometry from the implementation:
      tf0  = sr // window               # 1000 frames per second at sr=16000, window=16
      start = x_pad * tf0               # frame index where the user f0 takes over
      delta_t = round((t_max - t_min) * tf0 + 1)
      replace = np.interp(range(delta_t), inp_f0[:, 0] * 100, inp_f0[:, 1])
      f0[start : start + delta_t] = replace[:slice_len]
    """

    def test_splice_replaces_expected_slice(self):
        x_pad = 1
        tf0 = 16000 // 16  # 1000
        start = x_pad * tf0

        # Build f0 long enough to hold a splice well past the start offset.
        f0_raw = np.full(start + 200, 300.0)
        pipe = _make_pipeline(f0_raw, x_pad=x_pad)

        # Two control points: (t=0.00, f0=200), (t=0.05, f0=400). After * 100 the x-coords are [0, 5] (frames),
        # so np.interp over range(delta_t) covers 51 frames.
        inp_f0 = np.array(
            [
                [0.00, 200.0],
                [0.05, 400.0],
            ]
        )
        delta_t = int(np.round((0.05 - 0.00) * tf0 + 1))  # 51

        # Reference: interp on [0, 5] (the * 100 mapping) across range(51).
        expected_replace = np.interp(np.arange(delta_t), inp_f0[:, 0] * 100, inp_f0[:, 1])

        _, f0bak = pipe.get_f0(x=np.zeros(16000), f0_up_key=0, inp_f0=inp_f0)

        # Inside the splice window: matches the interpolated user f0.
        np.testing.assert_allclose(f0bak[start : start + delta_t], expected_replace)
        # Before and after the splice: the raw f0 (since f0_up_key=0) is preserved.
        np.testing.assert_allclose(f0bak[:start], 300.0)
        np.testing.assert_allclose(f0bak[start + delta_t :], 300.0)

    def test_splice_then_shift_does_not_re_shift_user_f0(self):
        # The implementation applies the shift first, then splices. So the user-provided f0 values appear in f0bak
        # exactly as given, regardless of `f0_up_key`. This pins that ordering: changing it would silently warp
        # user-provided pitch curves.
        x_pad = 1
        tf0 = 16000 // 16
        start = x_pad * tf0

        f0_raw = np.full(start + 100, 300.0)
        inp_f0 = np.array([[0.00, 200.0], [0.02, 200.0]])  # flat 200 Hz region
        delta_t = int(np.round(0.02 * tf0 + 1))  # 21 frames

        pipe = _make_pipeline(f0_raw, x_pad=x_pad)
        _, f0bak = pipe.get_f0(x=np.zeros(16000), f0_up_key=12, inp_f0=inp_f0)

        # User f0 is preserved verbatim (not multiplied by 2).
        np.testing.assert_allclose(f0bak[start : start + delta_t], 200.0)
        # Outside the splice, the shift still applies.
        np.testing.assert_allclose(f0bak[:start], 600.0)

    def test_splice_truncates_when_target_slice_is_shorter(self):
        # The implementation uses `shape = f0[start:start+len].shape[0]` and writes `replace[:shape]` so a long
        # interpolation gracefully truncates at the array end without raising.
        x_pad = 1
        tf0 = 16000 // 16
        start = x_pad * tf0

        # Only 10 frames of room past `start`.
        f0_raw = np.full(start + 10, 300.0)
        inp_f0 = np.array([[0.00, 200.0], [0.05, 400.0]])  # would want 51 frames; truncates to 10.

        pipe = _make_pipeline(f0_raw, x_pad=x_pad)
        _, f0bak = pipe.get_f0(x=np.zeros(16000), f0_up_key=0, inp_f0=inp_f0)

        expected_replace = np.interp(np.arange(51), inp_f0[:, 0] * 100, inp_f0[:, 1])[:10]
        np.testing.assert_allclose(f0bak[start:], expected_replace)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])