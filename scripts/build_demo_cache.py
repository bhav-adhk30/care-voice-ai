"""
Precompute CareVoicePipeline.process() for every clip in app/demo_clips/ into
app/cached_results.json, for the app's cached mode (CLAUDE.md section 9).

    HF_HUB_OFFLINE=1 python scripts/build_demo_cache.py

Rerun after any change to the models, thresholds or clips: the cache is a
replay of the pipeline, so a stale one shows numbers the system no longer gives.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import CareVoicePipeline  # noqa: E402

CONFIG = {
    "demo_dir": ROOT / "app" / "demo_clips",
    "out": ROOT / "app" / "cached_results.json",
    "audio_suffixes": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"},
}


def main() -> int:
    clips = sorted(p for p in CONFIG["demo_dir"].iterdir() if p.suffix.lower() in CONFIG["audio_suffixes"])
    if not clips:
        print(f"no clips in {CONFIG['demo_dir']}")
        return 1
    pipe = CareVoicePipeline(model_dir=str(ROOT / "models"))
    out = {}
    for p in clips:
        r = pipe.process(p)
        r["audio_path"] = p.name
        out[p.name] = r
        print(f"{p.name:40s} {r['request_class']:12s} P{r['base_priority']}->P{r['final_priority']}  {r['reason']}")
    CONFIG["out"].write_text(json.dumps(out, indent=2))
    print(f"\nwrote {len(out)} results to {CONFIG['out'].relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
