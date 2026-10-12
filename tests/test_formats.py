"""
Every upload format the app accepts must decode through both loaders.

The app's allow-list (app/streamlit_app.py CONFIG["audio_types"]) promises these
formats; this test keeps the promise honest. Clips are generated from a fixture with
ffmpeg at test time, including an .mp4 with a video track (its audio is used).
"""

import shutil
import subprocess

import pytest

from src.preprocess import load_audio, load_audio_acoustic
from tests.conftest import FIXTURES

FFMPEG = shutil.which("ffmpeg")
SRC = FIXTURES / "tts_chest_pain.wav"

# suffix -> extra ffmpeg output args
FORMATS = {
    "m4a": ["-c:a", "aac"], "mp4": ["-c:a", "aac"], "mov": ["-c:a", "aac"],
    "qta": ["-c:a", "aac", "-f", "mov"], "aac": ["-c:a", "aac"],
    "mp3": ["-c:a", "libmp3lame"], "flac": [], "ogg": ["-c:a", "libvorbis"],
    "opus": ["-c:a", "libopus"], "webm": ["-c:a", "libopus"], "aiff": [],
    "caf": [], "3gp": ["-ar", "8000", "-c:a", "aac"],
    "amr": ["-ar", "8000", "-ac", "1", "-c:a", "amr_nb"],
}


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not on PATH")
@pytest.mark.parametrize("suffix", list(FORMATS) + ["mp4-video"])
def test_format_decodes(suffix, tmp_path):
    if suffix == "mp4-video":
        out = tmp_path / "clip_video.mp4"
        cmd = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=3.5",
               "-i", str(SRC), "-shortest", "-c:v", "libx264", "-c:a", "aac", str(out)]
    else:
        out = tmp_path / f"clip.{suffix}"
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(SRC), *FORMATS[suffix], str(out)]
    if subprocess.run(cmd).returncode != 0:
        pytest.skip(f"this ffmpeg build cannot encode {suffix}")
    for loader in (load_audio, load_audio_acoustic):
        wav = loader(out)
        assert 3.0 < len(wav) / 16000 < 4.1, (suffix, loader.__name__, len(wav) / 16000)


def test_app_allow_list_covers_tested_formats():
    import re
    from tests.conftest import ROOT
    src = (ROOT / "app" / "streamlit_app.py").read_text()
    allowed = set(re.findall(r'"([a-z0-9]+)"', src[src.index('"audio_types"'):src.index("]", src.index('"audio_types"'))]))
    assert set(FORMATS) <= allowed, f"tested but not uploadable: {set(FORMATS) - allowed}"
