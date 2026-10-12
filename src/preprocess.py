"""
The single shared audio preprocessing module (CLAUDE.md section 2.5).

Every clip from every corpus (Medical Speech, CREMA-D, VIVAE, MUSAN, and any live demo
recording) passes through `load_audio`. Noise/codec augmentation goes through `augment`.
Never reimplement any of this inline in a notebook, job script, or the app: the
cross-corpus design depends on all corpora sharing the same sample rate, channel layout
and loudness, and drift between them fails silently.

Determinism
-----------
- `load_audio` has no randomness. Same file + same arguments -> bit-identical output on the
  same library versions (resampling uses soxr, which is deterministic).
- `augment` takes an explicit integer `seed`; noise choice and noise offset are drawn only
  from `np.random.default_rng(seed)`. Use `stable_seed(...)` to derive a per-clip seed from
  e.g. the clip path and SNR. Do NOT use Python's built-in `hash()` for this: it is salted
  per process, so seeds would change between Kaggle sessions.
- MP3 re-encoding shells out to ffmpeg/LAME, which is deterministic for identical input,
  bitrate and ffmpeg version. Log the ffmpeg version alongside cached features.

Dependencies: numpy, soundfile, librosa (soxr backend), pyloudnorm, and an `ffmpeg` binary
with libmp3lame on PATH (only for MP3 re-encoding).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from typing import Sequence

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf

CONFIG = {
    "target_sr": 16000,          # Hz, mono. Matches Whisper and WavLM input.
    "target_lufs": -23.0,        # EBU R128 integrated loudness target.
    "resample_type": "soxr_hq",  # deterministic, high quality
    "min_loudness_seconds": 0.4, # pyloudnorm's gating block; shorter clips are tiled to measure
    "max_gain_db": 40.0,         # cap on loudness gain, so near-silent clips are not blown up into pure noise floor
    "peak_ceiling": 0.99,        # after gain, scale down if any sample exceeds this (prevents hard clipping)
    "silence_floor_lufs": -70.0, # below this a clip is treated as silence and left unscaled
    "active_frame_ms": 25,       # frame length for speech-power estimation in SNR mixing
    "active_threshold_db": -40.0,# frames this far below the loudest frame are ignored for SNR power
    "ffmpeg_bin": "ffmpeg",
    # DeepFilterNet denoising: specified in the design, NOT evaluated, OFF.
    # deepfilterlib (its native core, latest 0.5.6) ships wheels only for Python
    # 3.8-3.11; this project runs 3.12 and building from source needs Rust. Every
    # number in results/ was measured without denoising. It used to default to on
    # and silently return its input when the import failed - a no-op claiming to
    # be on - so the default is now explicit, CareVoicePipeline logs it at
    # startup, and enabling it without DeepFilterNet installed raises.
    "enable_denoise": False,
    "enable_trim": True,
}

DENOISE_DISABLED_REASON = "deepfilterlib has no wheel for Python 3.12"


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

class AudioLoadError(Exception):
    """The file could not be read as audio. The message is shown to the nurse and
    logged, so it says what was wrong in plain words."""


# Formats libsndfile reads natively. If one of these fails to decode, the file is
# corrupt or not audio: fail at once rather than handing it to audioread, which
# retries every backend (~3 s) and then fails anyway.
_SOUNDFILE_NATIVE = {".wav", ".wave", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".aifc",
                     ".au", ".caf", ".mp3", ".w64", ".rf64"}


def _read(path: str) -> tuple[np.ndarray, int]:
    """Read any audio file as float64 (n_samples, n_channels) plus its native sample rate.

    Raises AudioLoadError for a missing, empty or undecodable file.
    """
    if not os.path.exists(path):
        raise AudioLoadError(f"file not found: {os.path.basename(path)}")
    if os.path.getsize(path) == 0:
        raise AudioLoadError(f"file is empty (0 bytes): {os.path.basename(path)}")
    try:
        data, sr = sf.read(path, dtype="float64", always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as e:
        if os.path.splitext(path)[1].lower() in _SOUNDFILE_NATIVE:
            # libsndfile prefixes the full path ("Error opening '/tmp/...': ");
            # keep only the cause, the nurse sees this string.
            cause = str(e).rsplit("': ", 1)[-1].strip().rstrip(".")
            raise AudioLoadError(f"not a readable audio file: {os.path.basename(path)} "
                                 f"({cause})") from e
        # Containers libsndfile cannot open at all (.m4a voice memos, .webm from a
        # browser mic): fall back to librosa (audioread/ffmpeg).
        try:
            data, sr = librosa.load(path, sr=None, mono=False, dtype=np.float64)
        except Exception as e2:
            raise AudioLoadError(f"could not decode {os.path.basename(path)} "
                                 f"({type(e2).__name__})") from e2
        data = np.atleast_2d(data).T  # librosa returns (channels, samples)
    if data.size == 0:
        raise AudioLoadError(f"file contains no audio samples: {os.path.basename(path)}")
    return data, int(sr)


def to_mono(data: np.ndarray) -> np.ndarray:
    """Average channels. Accepts (n,) or (n, channels); returns (n,)."""
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int = CONFIG["target_sr"]) -> np.ndarray:
    """Resample a mono signal with soxr. No-op when rates already match."""
    if orig_sr == target_sr:
        return wav
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr,
                            res_type=CONFIG["resample_type"])


def measure_lufs(wav: np.ndarray, sr: int = CONFIG["target_sr"]) -> float:
    """Integrated loudness (LUFS, ITU-R BS.1770) of a mono signal.

    pyloudnorm needs at least one 400 ms gating block. Shorter clips (e.g. brief VIVAE
    vocalisations) are tiled up to that length for measurement only; tiling preserves the
    signal's loudness, unlike zero-padding. Returns -inf for digital silence.
    """
    min_len = int(np.ceil(CONFIG["min_loudness_seconds"] * sr)) + 1
    if len(wav) == 0:
        return float("-inf")
    if len(wav) < min_len:
        wav = np.resize(wav, min_len)  # np.resize repeats the array cyclically
    meter = pyloudnorm.Meter(sr)
    return float(meter.integrated_loudness(wav))


def normalize_loudness(wav: np.ndarray, sr: int = CONFIG["target_sr"],
                       target_lufs: float = CONFIG["target_lufs"]) -> np.ndarray:
    """Scale a mono signal to `target_lufs`, with a gain cap and peak protection.

    - Silence (loudness below `silence_floor_lufs`, or -inf) is returned unchanged.
    - Gain is capped at `max_gain_db` so very quiet clips are raised but their noise floor
      is not amplified without bound.
    - If the gained signal peaks above `peak_ceiling`, the whole clip is scaled down to that
      ceiling. Such clips (already loud or clipped at source) end slightly below target;
      that is preferred over introducing new clipping.
    """
    loudness = measure_lufs(wav, sr)
    if not np.isfinite(loudness) or loudness < CONFIG["silence_floor_lufs"]:
        return wav
    gain_db = min(target_lufs - loudness, CONFIG["max_gain_db"])
    out = wav * (10.0 ** (gain_db / 20.0))
    peak = np.max(np.abs(out))
    if peak > CONFIG["peak_ceiling"]:
        out = out * (CONFIG["peak_ceiling"] / peak)
    return out


def pad_or_crop(wav: np.ndarray, n_samples: int) -> np.ndarray:
    """Force a signal to exactly `n_samples`.

    Crops from the START (keeps the onset, where requests and vocalisations begin) and
    zero-pads at the END. Deterministic; no random crop.
    """
    if len(wav) >= n_samples:
        return wav[:n_samples]
    return np.pad(wav, (0, n_samples - len(wav)))


def load_audio(path, target_sr=16000, target_lufs=-23.0, max_seconds=None,
               return_meta: bool = False):
    """Load → mono → resample to 16 kHz → loudness normalise → optional pad/crop.
    Returns float32 numpy array. Every corpus goes through this. No exceptions.

    Parameters
    ----------
    path : str or os.PathLike
        Any format soundfile can read (WAV, FLAC, OGG, and MP3 on libsndfile >= 1.1);
        other formats fall back to librosa.
    target_sr : int
        Output sample rate in Hz.
    target_lufs : float or None
        Integrated loudness target. Measured over the whole clip BEFORE any crop/pad, so
        the gain does not depend on `max_seconds`. Pass None to skip normalisation (only
        for diagnostics; never for data that feeds a model).
    max_seconds : float or None
        If given, the output is exactly `round(max_seconds * target_sr)` samples: cropped
        from the start or zero-padded at the end. If None, natural length is kept.

    return_meta : bool
        If True, also return a dict of raw (pre-normalisation) signal statistics for
        `energy_present`. Default False, so existing callers are unaffected.

    Returns
    -------
    np.ndarray, shape (n_samples,), dtype float32, nominally in [-1, 1].
    With return_meta=True: (wav, {"raw_rms", "raw_peak", "raw_duration_s"}).

    Notes
    -----
    Denoising (DeepFilterNet) and silence trimming are not applied here. If they are added
    they belong in this module, applied identically to every corpus.
    """
    data, sr = _read(os.fspath(path))
    wav = to_mono(data)
    wav = resample(wav, sr, target_sr)
    # Measured before normalisation. After -23 LUFS the gain applied to a
    # near-silent clip is enormous, so post-normalisation level carries no
    # information about whether anyone actually spoke.
    raw_rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
    raw_peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    raw_dur = len(wav) / target_sr
    if target_lufs is not None:
        wav = normalize_loudness(wav, target_sr, target_lufs)
    if max_seconds is not None:
        wav = pad_or_crop(wav, int(round(max_seconds * target_sr)))
    wav = wav.astype(np.float32)
    if return_meta:
        return wav, {"raw_rms": raw_rms, "raw_peak": raw_peak,
                     "raw_duration_s": raw_dur}
    return wav


# --------------------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------------------

def stable_seed(*parts) -> int:
    """Derive a reproducible 63-bit seed from arbitrary values (e.g. clip path, SNR).

    Uses SHA-256 of the parts' string forms, so it is identical across processes,
    machines and Python versions (unlike built-in `hash()`).

        seed = stable_seed("cremad/1001_DFA_ANG_XX.wav", 10, "musan")
    """
    key = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 1


def load_noise_bank(paths: Sequence, target_sr: int = CONFIG["target_sr"]) -> list[np.ndarray]:
    """Load noise files (e.g. MUSAN noise/music) through `load_audio`.

    Order is the order of `paths`; sort them first if they come from a directory listing,
    otherwise noise selection in `augment` is not reproducible across filesystems.
    """
    return [load_audio(p, target_sr=target_sr) for p in paths]


def _active_power(wav: np.ndarray, sr: int) -> float:
    """Mean power over frames within `active_threshold_db` of the loudest frame.

    Used as the "signal" power for SNR so that leading/trailing silence and zero padding do
    not make the speech look quieter than it is (which would under-scale the noise).
    """
    frame = max(1, int(sr * CONFIG["active_frame_ms"] / 1000))
    n = len(wav) // frame
    if n == 0:
        return float(np.mean(wav ** 2))
    frame_pow = np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1)
    peak = frame_pow.max()
    if peak <= 0:
        return 0.0
    active = frame_pow >= peak * 10.0 ** (CONFIG["active_threshold_db"] / 10.0)
    return float(frame_pow[active].mean())


def mix_at_snr(wav: np.ndarray, noise: np.ndarray, snr_db: float,
               sr: int = CONFIG["target_sr"]) -> np.ndarray:
    """Add `noise` (same length as `wav`) scaled so that active-speech SNR equals `snr_db`."""
    sig_pow = _active_power(wav, sr)
    noise_pow = float(np.mean(noise ** 2))
    if sig_pow <= 0 or noise_pow <= 0:
        return wav.copy()
    scale = np.sqrt(sig_pow / (noise_pow * 10.0 ** (snr_db / 10.0)))
    return wav + scale * noise


def mp3_roundtrip(wav: np.ndarray, bitrate_kbps: int,
                  sr: int = CONFIG["target_sr"]) -> np.ndarray:
    """Encode to MP3 with LAME at `bitrate_kbps` (CBR) and decode back.

    Simulates the codec damage of phone/intercom/compressed recordings. Uses temporary
    files rather than pipes so ffmpeg can write the LAME gapless header, which lets the
    decoder strip encoder delay; the result is then trimmed/padded to the input length
    so it stays sample-aligned with the original.
    """
    ffmpeg = shutil.which(CONFIG["ffmpeg_bin"])
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; required for MP3 re-encoding")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        enc = os.path.join(tmp, "enc.mp3")
        sf.write(src, wav.astype(np.float32), sr, subtype="FLOAT")
        common = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        subprocess.run(common + ["-i", src, "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k",
                                 "-ac", "1", "-ar", str(sr), "-bitexact", enc],
                       check=True)
        out = subprocess.run(common + ["-i", enc, "-f", "f32le", "-ac", "1", "-ar", str(sr),
                                       "pipe:1"],
                             check=True, capture_output=True).stdout
    dec = np.frombuffer(out, dtype=np.float32).astype(np.float64)
    return pad_or_crop(dec, len(wav))


def augment(wav, noise_bank, snr_db, *, seed: int, mp3_bitrate_kbps: int | None = None,
            sr: int = CONFIG["target_sr"], target_lufs: float = CONFIG["target_lufs"]):
    """Corrupt a clip with additive noise and/or MP3 re-encoding, deterministically.

    Order of operations mirrors a real capture chain:
        1. add a noise segment at `snr_db` (acoustic noise in the room)
        2. MP3 encode/decode at `mp3_bitrate_kbps` (codec in the transmission path)
        3. re-normalise loudness to `target_lufs`
    Step 3 matters: a real noisy recording would pass through `load_audio`, which
    normalises the mixture. Augmented training data must look the same.

    Parameters
    ----------
    wav : np.ndarray
        Mono clip already produced by `load_audio` (so at `sr`, loudness normalised).
    noise_bank : sequence of np.ndarray
        Noise clips at `sr`, e.g. from `load_noise_bank`. Ignored when `snr_db` is None.
    snr_db : float or None
        Target signal-to-noise ratio in dB, measured against active speech frames (see
        `_active_power`). None (or +inf) skips noise mixing, e.g. for MP3-only augmentation.
    seed : int, keyword-only, required
        Controls which noise clip is used and where in it the segment starts. Required so
        that no call is accidentally non-reproducible. Derive per clip with `stable_seed`.
    mp3_bitrate_kbps : int or None
        If given, apply an MP3 round trip at this CBR bitrate (e.g. 16, 32, 64).
    sr : int
        Sample rate of `wav` and of every noise clip.
    target_lufs : float or None
        Loudness to re-normalise to after corruption. None skips it.

    Returns
    -------
    np.ndarray float32, same length as `wav`.

    Noise selection: one clip is chosen uniformly from `noise_bank`. If it is longer than
    `wav`, a contiguous segment starts at a uniform random offset; if shorter, it is tiled.
    """
    wav = np.asarray(wav, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = wav

    if snr_db is not None and np.isfinite(snr_db):
        if len(noise_bank) == 0:
            raise ValueError("noise_bank is empty but snr_db was given")
        noise = np.asarray(noise_bank[int(rng.integers(len(noise_bank)))], dtype=np.float64)
        if len(noise) == 0:
            raise ValueError("noise_bank contains an empty clip")
        if len(noise) >= len(wav):
            start = int(rng.integers(len(noise) - len(wav) + 1))
            noise = noise[start:start + len(wav)]
        else:
            noise = np.resize(noise, len(wav))  # cyclic tiling
        out = mix_at_snr(out, noise, float(snr_db), sr)

    if mp3_bitrate_kbps is not None:
        out = mp3_roundtrip(out, int(mp3_bitrate_kbps), sr)

    if target_lufs is not None:
        out = normalize_loudness(out, sr, target_lufs)
    else:
        peak = np.max(np.abs(out)) if len(out) else 0.0
        if peak > CONFIG["peak_ceiling"]:
            out = out * (CONFIG["peak_ceiling"] / peak)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Denoise and trim.
#
# Deliberately NOT inside load_audio(). At inference the system sees noisy ward
# audio and denoises it, so training must see the same order:
#     load_audio -> augment (add noise) -> denoise -> trim -> features
# Denoising the already-clean corpora before augmentation would mean training
# and inference differ, and the measurements would be wrong.
# ---------------------------------------------------------------------------

_DF_MODEL = None
_DF_STATE = None


def _init_deepfilternet():
    """Load DeepFilterNet once and cache it. Raises if it cannot be loaded: denoising
    that was asked for must happen or fail loudly, never silently pass audio through."""
    global _DF_MODEL, _DF_STATE
    if _DF_MODEL is None:
        try:
            from df.enhance import init_df
        except ImportError as e:
            raise RuntimeError(
                "CONFIG['enable_denoise'] is True but DeepFilterNet is not installed "
                f"({e}). {DENOISE_DISABLED_REASON}; set enable_denoise to False.") from e
        _DF_MODEL, _DF_STATE, _ = init_df()
    return _DF_MODEL, _DF_STATE


def denoise(wav, sr=16000):
    """
    DeepFilterNet3 denoising. Same sample rate and length in and out.

    DeepFilterNet runs natively at 48 kHz, so the signal is resampled up,
    enhanced, and resampled back.

    Disabled by default (CONFIG["enable_denoise"], see the comment there): then the
    input is returned unchanged, and CareVoicePipeline has said so at startup. When
    enabled, a missing DeepFilterNet raises rather than passing audio through.
    """
    if not CONFIG["enable_denoise"]:
        return wav

    model, state = _init_deepfilternet()

    import torch
    from df.enhance import enhance

    n_in = len(wav)
    wav48 = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=48000)
    enhanced = enhance(model, state, torch.from_numpy(wav48).unsqueeze(0))
    out = librosa.resample(enhanced.squeeze(0).cpu().numpy(), orig_sr=48000, target_sr=sr)

    if len(out) < n_in:
        out = np.pad(out, (0, n_in - len(out)))
    return out[:n_in].astype(np.float32)


def trim(wav, sr=16000, top_db=None, keep_pad_ms=None):
    """
    Trim leading and trailing silence, keeping a short margin so the first and
    last phonemes are not clipped. Deterministic.

    Returns the input unchanged if trimming would leave almost nothing, which
    happens on very quiet clips and would otherwise destroy the sample.
    """
    if not CONFIG["enable_trim"]:
        return wav

    top_db = top_db if top_db is not None else CONFIG.get("trim_top_db", 30)
    keep_ms = keep_pad_ms if keep_pad_ms is not None else CONFIG.get("trim_keep_pad_ms", 50)

    if len(wav) == 0 or not np.any(wav):
        return wav

    _, (start, end) = librosa.effects.trim(wav, top_db=top_db)
    pad = int(sr * keep_ms / 1000)
    start = max(0, start - pad)
    end = min(len(wav), end + pad)

    out = wav[start:end]
    if len(out) < int(0.1 * sr):      # under 100 ms left: trimming went wrong
        return wav
    return out


# ---------------------------------------------------------------------------
# Acoustic-branch loader
#
# There are two loaders, and that is section 2.5 satisfied rather than violated.
# The rule exists so that every clip reaching a model has been prepared exactly
# as that model's training data was. The two branches were trained differently:
#
#   load_audio           -23 LUFS, natural length. Whisper (pretrained, any level)
#                        and the ASR_MIN measurement use this.
#   load_audio_acoustic  RMS 0.05, gain capped at 10x, peak 0.99, crop to 4 s,
#                        never pad. Bit-for-bit the loader the acoustic heads were
#                        fitted on in notebooks/11_head_refit.ipynb (load_norm),
#                        recorded in the joblibs as extraction contract
#                        "no_pad_meanpool_rms0.05". WavLM has do_normalize=False,
#                        so input level reaches the features directly; feeding the
#                        heads LUFS-normalised audio would move their scores.
#
# No denoise and no trim on this path: notebook 11 applied neither, so the
# heads have never seen trimmed or denoised audio. Both definitions live here,
# next to each other, and nowhere else.
# ---------------------------------------------------------------------------

ACOUSTIC_CONTRACT = "no_pad_meanpool_rms0.05"

ACOUSTIC_CONFIG = {
    "target_rms": 0.05,
    "max_gain": 10.0,
    "peak_ceiling": 0.99,
    "rms_floor": 1e-8,       # below this the clip is left unscaled, as in training
    "max_seconds": 4.0,      # crop from the start; shorter clips keep their length
}


def load_audio_acoustic(path, target_sr: int = 16000, cfg: dict = ACOUSTIC_CONFIG) -> np.ndarray:
    """Load for the WavLM branch under contract ACOUSTIC_CONTRACT.

    librosa load at 16 kHz mono -> RMS-normalise to 0.05 (gain capped at 10x)
    -> peak-limit to 0.99 -> crop to 4 s. Never pads: a short clip stays short and
    is mean-pooled over its own frames. Returns float32.
    """
    wav, _ = librosa.load(os.fspath(path), sr=target_sr, mono=True)
    rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
    if rms > cfg["rms_floor"]:
        wav = wav * min(cfg["target_rms"] / rms, cfg["max_gain"])
        peak = float(np.abs(wav).max())
        if peak > cfg["peak_ceiling"]:
            wav = wav * (cfg["peak_ceiling"] / peak)
    return wav[: int(round(cfg["max_seconds"] * target_sr))].astype(np.float32)


# ---------------------------------------------------------------------------
# Native-rate loader for the prosodic distress feature
#
# The third loader, and the one that must NOT resample. Harmonics-to-noise ratio
# (src/prosody.py) changes by 4-8 dB when 48 kHz audio is resampled to 16 kHz,
# and for some speakers the calm-vs-distress difference disappears entirely
# (Bhavishya, chest: +3.4 dB at 48 kHz, -0.2 dB at 16 kHz). So HNR is computed on
# the audio at its native rate, before anything downsamples it for Whisper or
# WavLM. No loudness normalisation either: HNR is a ratio, level-invariant.
# ---------------------------------------------------------------------------


def load_audio_native(path) -> tuple[np.ndarray, int]:
    """Mono float32 at the file's own sample rate, untouched otherwise.
    Raises AudioLoadError like load_audio."""
    data, sr = _read(os.fspath(path))
    return to_mono(data).astype(np.float32), sr


# ---------------------------------------------------------------------------
# Energy pre-filter for the speech guard
#
# The speech decision itself is the arousal head's non_speech class (rules.py,
# NON_SPEECH_HIGH), which was trained on ward noise and silence. This check only
# saves running WavLM on clips with essentially no signal, so it is set at
# digital-silence level: it must never be the reason a quiet real call is lost.
# ---------------------------------------------------------------------------

SPEECH_GUARD = {
    # Pre-normalisation RMS, about -80 dBFS. Deliberately far below any real
    # recording; everything above it is decided by the non_speech class.
    "MIN_RAW_RMS": 1.0e-4,
}


def energy_present(meta: dict, cfg: dict = SPEECH_GUARD) -> tuple[bool, str]:
    """
    Cheap pre-filter: is there any signal at all? Returns (present, reason).
    `meta` is the dict from load_audio(return_meta=True).

    Energy only, never the transcript - a wordless cry has no transcript and
    must still reach the acoustic branch.
    """
    if meta["raw_rms"] < cfg["MIN_RAW_RMS"]:
        return False, (f"signal level below floor "
                       f"(RMS {meta['raw_rms']:.2e} < {cfg['MIN_RAW_RMS']:.0e})")
    return True, ""
