"""
Download the two pretrained (not trained-here) models into models/ so the pipeline and
the demo run fully offline with HF_HUB_OFFLINE=1 (CLAUDE.md section 9, demo reliability).

    python scripts/fetch_models.py            # fetch anything missing
    python scripts/fetch_models.py --force    # re-download everything

Whisper small
    faster-whisper needs CTranslate2 weights, not the transformers checkpoint, so this
    fetches Systran/faster-whisper-small: a direct CTranslate2 conversion of
    openai/whisper-small (same 244 M weights, float16 on disk). The pipeline quantises to
    int8 at load time via compute_type="int8", so no separate int8 copy is needed.

WavLM Base+
    microsoft/wavlm-base-plus, config + feature-extractor config + weights only. Used
    FROZEN (CLAUDE.md section 2.2).

Run once, with internet, before going anywhere the network cannot be trusted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "model_dir": ROOT / "models",
    "models": {
        # local folder name -> (repo id, files that must exist afterwards)
        "faster-whisper-small": (
            "Systran/faster-whisper-small",
            ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"],
        ),
        "wavlm-base-plus": (
            "microsoft/wavlm-base-plus",
            ["config.json", "preprocessor_config.json"],
        ),
    },
    # Never download README/.gitattributes/TF/Flax weights; disk is tight.
    "skip_suffixes": (".md", ".gitattributes", ".h5", ".msgpack", ".onnx", ".ot"),
}


def _pick_files(repo_id: str) -> list[str]:
    """Files to fetch from a repo: configs, tokenizer files and ONE copy of the weights
    (safetensors preferred over pytorch_model.bin when both exist)."""
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(repo_id)
             if not f.endswith(CONFIG["skip_suffixes"]) and "/" not in f]
    if any(f.endswith(".safetensors") for f in files):
        files = [f for f in files if not f.endswith(".bin") or f == "model.bin"]
    return files


def fetch(name: str, repo_id: str, required: list[str], force: bool) -> Path:
    from huggingface_hub import hf_hub_download

    dst = CONFIG["model_dir"] / name
    if not force and dst.is_dir() and all((dst / f).exists() for f in required):
        print(f"[skip] {name}: already present at {dst}")
        return dst
    if force and dst.exists():
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)
    files = _pick_files(repo_id)
    print(f"[get]  {repo_id} -> {dst}  ({', '.join(files)})")
    for f in files:
        hf_hub_download(repo_id, f, local_dir=dst)

    missing = [f for f in required if not (dst / f).exists()]
    if not any((dst / f).exists() for f in ("model.bin", "model.safetensors", "pytorch_model.bin")):
        missing.append("<weights>")
    if missing:
        raise RuntimeError(f"{name}: download incomplete, missing {missing}")
    # hf_hub_download leaves a small metadata cache inside local_dir; not needed offline.
    shutil.rmtree(dst / ".cache", ignore_errors=True)
    size_mb = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1e6
    print(f"[ok]   {name}: {size_mb:.0f} MB")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        print("HF_HUB_OFFLINE=1 is set; unset it to download.", file=sys.stderr)
        return 1

    for name, (repo_id, required) in CONFIG["models"].items():
        fetch(name, repo_id, required, args.force)
    print("\nAll models in place. The pipeline can now run with HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
