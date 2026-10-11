"""
CareVoicePipeline: the one entry point (CLAUDE.md section 2.7).

The Streamlit app and the end-to-end evaluation both import this class. Never write a
second implementation of the pipeline anywhere else.

    audio -> load_audio (-23 LUFS) -> energy pre-filter
          -> load_audio_acoustic (RMS 0.05, crop 4 s, no pad) -> WavLM Base+ (frozen,
             mean-pooled over real frames) -> arousal head (fear, non_speech) + distress gate
          -> denoise -> trim -> Whisper small (faster-whisper int8)
          -> speech guard (non_speech AND no trusted transcript AND no distress)
          -> if speech and a transcript: ClinicalBERT
          -> apply_rules (src/rules.py) -> final priority + reason

The two branches share nothing but the source file: each has its own loader,
matching how its model was trained (see preprocess.py). They meet only in
apply_rules, which is a deterministic rule and not a trained model (section 2.1).
Both branches run before the speech guard, which needs an output from each.

Why both branches exist: text alone misses 26% of "Heart hurts" and 24% of "Hard to
breath" clips, because cardiac and respiratory complaints are phrased in pain vocabulary.
The acoustic branch is there to recover that measured gap. Whether it does so end to end
cannot be measured without jointly labelled clips, so nothing here claims it does.

CLI, for a smoke test:
    python -m src.pipeline path/to/clip.wav --bed 14
"""

from __future__ import annotations

import json
import os
import random
import time
import warnings
from pathlib import Path

# Everything loads from models/. Never let a failed network call hang the demo.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import joblib
import numpy as np
import torch

from src.mapping import class_base_priority
from src.preprocess import (ACOUSTIC_CONTRACT, denoise, energy_present, load_audio,
                            load_audio_acoustic, trim)
from src.rules import CONFIG as RULES_CONFIG
from src.rules import UNCLASSIFIED, apply_rules, speech_guard

CONFIG = {
    "sr": 16000,
    "seed": 42,

    # Subfolders / files of model_dir. Whisper and WavLM come from scripts/fetch_models.py.
    "whisper_dir": "faster-whisper-small",
    "wavlm_dir": "wavlm-base-plus",
    "bert_dir": "clinicalbert_requestclass",
    "arousal_head": "arousal_head.joblib",
    "distress_gate": "distress_gate.joblib",

    # ASR. int8 on CPU is ~4x faster than the HF pipeline (CLAUDE.md section 9).
    "whisper_compute_type": "int8",
    # Decoder settings. ASR_MIN in rules.py is a percentile of the log-prob these
    # produce, so notebooks/09_threshold_recheck.ipynb measures it through
    # transcribe() below with exactly this dict. Change one, re-measure the other.
    "whisper_decode": {
        "language": "en",
        "beam_size": 5,
        # A single temperature disables faster-whisper's fallback ladder. A segment
        # retried at a higher temperature has a log-prob that is not comparable to a
        # first-pass one, which would make asr_confidence meaningless against ASR_MIN.
        "temperature": 0.0,
        "condition_on_previous_text": False,
        # Off: a VAD would drop a wordless groan before Whisper sees it. "No speech"
        # is decided by rules.speech_guard instead.
        "vad_filter": False,
    },
    # No speech decoded at all: treat the transcript as maximally untrusted. Any value
    # below rules.CONFIG["ASR_MIN"] works; this one is JSON-safe, unlike -inf.
    "empty_transcript_logprob": -10.0,

    # Text classifier.
    "bert_max_length": 128,

    # Acoustic branch, contract "no_pad_meanpool_rms0.05" (notebooks/11_head_refit).
    # No fixed window: each clip is fed at its own length (cropped to 4 s by
    # load_audio_acoustic) and mean-pooled over its real frames. The v1 heads
    # zero/noise-padded to 4 s and the padding carried the label (duration alone
    # separated the training corpora at AUC 0.9965). Only clips under this length
    # are padded, to give the encoder enough frames, exactly as in training.
    "acoustic_min_s": 0.2,

    # Run one dummy clip through every stage at load time. Without it the first real
    # call pays for numba JIT in trim() and first-call setup in Whisper and BERT
    # (~10 s measured on CPU), which is the call the audience sees.
    "warmup": True,
}


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


# Module-level so notebooks can measure thresholds through the exact code the
# pipeline runs, rather than a copy of it.

def transcribe(asr, wav: np.ndarray) -> tuple[str, float, list]:
    """
    Returns (transcript, asr_confidence, segments).

    asr_confidence is the mean log-probability over every decoded token: each
    segment's avg_logprob weighted by its token count. If nothing is decoded it is
    CONFIG["empty_transcript_logprob"], below ASR_MIN, so the transcript is untrusted.
    """
    segments, _ = asr.transcribe(wav, **CONFIG["whisper_decode"])
    segments = list(segments)  # the generator is what actually runs the decoder
    text = " ".join(s.text.strip() for s in segments).strip()
    if not text:
        return "", CONFIG["empty_transcript_logprob"], segments
    n_tok = np.array([max(len(s.tokens), 1) for s in segments], dtype=float)
    lp = np.array([s.avg_logprob for s in segments], dtype=float)
    return text, float((lp * n_tok).sum() / n_tok.sum()), segments


@torch.no_grad()
def embed_clip(feature_extractor, wavlm, wav: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Frozen WavLM Base+, last hidden state mean-pooled over the clip's own
    frames -> (1, 768). `wav` comes from preprocess.load_audio_acoustic.

    Batch size 1 at natural length, so every frame is real and no attention mask
    is needed (none is passed: group-norm WavLM was pretrained without one).
    Named embed_clip, not embed, so code written against the old padded embed()
    fails loudly instead of silently changing meaning."""
    min_len = int(round(CONFIG["acoustic_min_s"] * CONFIG["sr"]))
    if len(wav) < min_len:
        wav = np.pad(wav, (0, min_len - len(wav)))
    inp = feature_extractor(wav.astype(np.float32), sampling_rate=CONFIG["sr"],
                            return_tensors="pt")
    h = wavlm(inp.input_values.to(device)).last_hidden_state
    return h.mean(dim=1).float().cpu().numpy()


class CareVoicePipeline:
    def __init__(self, model_dir="models", device="cpu"):
        t0 = time.perf_counter()
        self.model_dir = Path(model_dir)
        self.device = device

        random.seed(CONFIG["seed"])
        np.random.seed(CONFIG["seed"])
        torch.manual_seed(CONFIG["seed"])

        self._load_asr()
        self._load_text()
        self._load_acoustic()
        if CONFIG["warmup"]:
            self._warmup()
        self.load_ms = _ms(t0)

    # ------------------------------------------------------------------ loading

    def _path(self, key: str) -> Path:
        p = self.model_dir / CONFIG[key]
        if not p.exists():
            hint = (" Run `python scripts/fetch_models.py`."
                    if key in ("whisper_dir", "wavlm_dir") else "")
            raise FileNotFoundError(f"Missing {p}.{hint}")
        return p

    def _load_asr(self):
        from faster_whisper import WhisperModel

        self.asr = WhisperModel(
            str(self._path("whisper_dir")),
            device="cuda" if self.device.startswith("cuda") else "cpu",
            compute_type=CONFIG["whisper_compute_type"],
            local_files_only=True,
        )

    def _load_text(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        bert_dir = self._path("bert_dir")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir, local_files_only=True)
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            bert_dir, local_files_only=True).to(self.device).eval()
        # Index order of the trained head. Not necessarily mapping.CLASSES order.
        self.classes = json.loads((bert_dir / "classes.json").read_text())["classes"]
        if self.bert.config.num_labels != len(self.classes):
            raise ValueError(f"classes.json has {len(self.classes)} classes, model has "
                             f"{self.bert.config.num_labels} outputs")

    def _load_acoustic(self):
        from transformers import AutoFeatureExtractor, WavLMModel

        wavlm_dir = self._path("wavlm_dir")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            wavlm_dir, local_files_only=True)
        self.wavlm = WavLMModel.from_pretrained(
            wavlm_dir, local_files_only=True).to(self.device).eval()
        for p in self.wavlm.parameters():
            p.requires_grad = False  # FROZEN, section 2.2

        arousal = joblib.load(self._path("arousal_head"))
        distress = joblib.load(self._path("distress_gate"))

        # The extraction contract travels with the heads. Refuse to run heads that
        # were fitted on differently prepared features: their scores would be
        # meaningless and nothing downstream could tell.
        if distress.get("extraction") != ACOUSTIC_CONTRACT:
            raise ValueError(f"distress_gate.joblib extraction={distress.get('extraction')!r}, "
                             f"pipeline implements {ACOUSTIC_CONTRACT!r}")
        if arousal.get("escalates", True) or "non_speech_index" not in arousal:
            raise ValueError("arousal_head.joblib is not a v2 head (needs escalates=False "
                             "and a non_speech class)")

        self.arousal_model = arousal["model"]
        self.arousal_classes = list(arousal["classes"])
        self.fear_index = int(arousal["fear_index"])
        self.non_speech_index = int(arousal["non_speech_index"])
        for idx, name in ((self.fear_index, "fear"), (self.non_speech_index, "non_speech")):
            if self.arousal_classes[idx] != name:
                raise ValueError(f"arousal head index {idx} is "
                                 f"{self.arousal_classes[idx]!r}, expected {name!r}")

        self.distress_model = distress["model"]
        self.distress_pos_index = list(self.distress_model.classes_).index(1)

        # rules.py is the single source of truth for thresholds. The copies saved with
        # the heads are only checked, never used.
        for name, saved, key in (("distress_gate", distress.get("threshold"), "DISTRESS_HIGH"),
                                 ("arousal_head", arousal.get("non_speech_threshold"), "NON_SPEECH_HIGH")):
            if saved is not None and abs(saved - RULES_CONFIG[key]) > 1e-9:
                warnings.warn(f"{name}.joblib threshold {saved} differs from "
                              f"rules.CONFIG[{key!r}]={RULES_CONFIG[key]}; using rules.py")

    def _warmup(self):
        rng = np.random.default_rng(CONFIG["seed"])
        wav = (0.05 * rng.standard_normal(CONFIG["sr"])).astype(np.float32)
        self._acoustic(wav)
        self._transcribe(trim(wav, sr=CONFIG["sr"]))
        self._classify("warm up")

    # ------------------------------------------------------------------ branches

    def _transcribe(self, wav: np.ndarray) -> tuple[str, float]:
        text, logprob, _ = transcribe(self.asr, wav)
        return text, logprob

    @torch.no_grad()
    def _classify(self, transcript: str) -> dict[str, float]:
        enc = self.tokenizer(transcript, return_tensors="pt", truncation=True,
                             max_length=CONFIG["bert_max_length"]).to(self.device)
        probs = torch.softmax(self.bert(**enc).logits, dim=-1)[0].cpu().numpy()
        return {c: float(p) for c, p in zip(self.classes, probs)}

    def _acoustic(self, wav: np.ndarray) -> dict:
        """Both heads on one unpadded embedding. Returns fear (display only),
        non_speech (the speech guard) and distress (the escalation signal)."""
        emb = embed_clip(self.feature_extractor, self.wavlm, wav, self.device)
        ap = self.arousal_model.predict_proba(emb)[0]
        return {
            "arousal": float(ap[self.fear_index]),
            "non_speech_prob": float(ap[self.non_speech_index]),
            "distress": float(self.distress_model.predict_proba(emb)[0, self.distress_pos_index]),
        }

    # ------------------------------------------------------------------ entry point

    def process(self, audio_path, bed=None) -> dict:
        """
        Run one clip through the whole system. Returns every intermediate, not just the
        final priority, so the dashboard can show WHY a call was escalated.

        arousal (fear probability) is reported for display only; it never affects
        priority. Acoustic values are None when the energy pre-filter stopped the
        clip before WavLM ran.
        """
        lat = {}
        t_all = time.perf_counter()

        t0 = time.perf_counter()
        wav, meta = load_audio(audio_path, target_sr=CONFIG["sr"], return_meta=True)
        lat["load"] = _ms(t0)
        duration_s = len(wav) / CONFIG["sr"]
        energy_ok, energy_reason = energy_present(meta)

        # --- acoustic branch ---
        acoustic = {"arousal": None, "non_speech_prob": None, "distress": None}
        lat["load_acoustic"] = lat["wavlm_and_heads"] = 0.0
        if energy_ok:
            t0 = time.perf_counter()
            wav_ac = load_audio_acoustic(audio_path, target_sr=CONFIG["sr"])
            lat["load_acoustic"] = _ms(t0)
            t0 = time.perf_counter()
            acoustic = self._acoustic(wav_ac)
            lat["wavlm_and_heads"] = _ms(t0)

        # --- ASR: every clip with any signal ---
        # Whisper's output is one of the three speech-guard conditions, so it runs
        # before the guard decides (~1.2 s on CPU, accepted).
        transcript, asr_confidence = "", None
        lat["denoise"] = lat["trim"] = lat["asr"] = 0.0
        if energy_ok:
            t0 = time.perf_counter()
            wav = denoise(wav, sr=CONFIG["sr"])
            lat["denoise"] = _ms(t0)
            t0 = time.perf_counter()
            wav = trim(wav, sr=CONFIG["sr"])
            lat["trim"] = _ms(t0)
            t0 = time.perf_counter()
            transcript, asr_confidence = self._transcribe(wav)
            lat["asr"] = _ms(t0)

        has_speech, speech_reason = speech_guard(
            energy_ok, acoustic["non_speech_prob"], transcript, asr_confidence,
            acoustic["distress"], energy_reason)
        # A no-speech call's transcript is by definition empty or an untrusted
        # hallucination ("Thank you." on fan noise). Kept for display, never
        # classified.
        heard = transcript
        if not has_speech:
            transcript = ""

        # --- text branch ---
        # ClinicalBERT only runs on a non-empty transcript: on "" it is out of
        # distribution (it predicted Emergency), so an empty transcript - no speech,
        # or a groan Whisper cannot decode - is UNCLASSIFIED at P_NO_SPEECH and the
        # distress gate still applies. UNCLASSIFIED is a pipeline sentinel, not a
        # trained class; the dashboard renders it as a grey badge.
        t0 = time.perf_counter()
        if transcript.strip():
            class_probs = self._classify(transcript)
            request_class = max(class_probs, key=class_probs.get)
            class_confidence = class_probs[request_class]
            base_priority = class_base_priority(request_class)
        else:
            request_class, class_probs, class_confidence = UNCLASSIFIED, {}, 0.0
            base_priority = RULES_CONFIG["P_NO_SPEECH"]
        lat["text_classifier"] = _ms(t0)

        # --- the rule ---
        t0 = time.perf_counter()
        decision = apply_rules(
            base_priority=base_priority,
            request_class=request_class,
            class_confidence=class_confidence,
            distress=acoustic["distress"] if acoustic["distress"] is not None else 0.0,
            asr_confidence=asr_confidence,
            energy_ok=energy_ok,
            non_speech_prob=acoustic["non_speech_prob"],
            energy_reason=energy_reason,
            transcript=transcript,
            asr_heard=heard,
        )
        lat["rules"] = _ms(t0)
        lat["total"] = _ms(t_all)

        return {
            "audio_path": str(audio_path),
            "bed": bed,
            "duration_s": round(duration_s, 3),
            "transcript": transcript,
            "asr_heard": heard,
            "asr_confidence": asr_confidence,
            "asr_trusted": decision.asr_trusted,
            "energy_ok": energy_ok,
            "non_speech_prob": acoustic["non_speech_prob"],
            "speech_present": has_speech,
            "speech_reason": speech_reason,
            "request_class": request_class,
            "class_probs": class_probs,
            "class_confidence": class_confidence,
            "base_priority": base_priority,
            "arousal": acoustic["arousal"],
            "distress": acoustic["distress"],
            "distress_fired": decision.distress_fired,
            "final_priority": decision.final_priority,
            "escalated": decision.escalated,
            "reason": decision.reason,
            "acoustic_contract": ACOUSTIC_CONTRACT,
            "latency_ms": lat,
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run one clip through CareVoicePipeline.")
    ap.add_argument("audio_path")
    ap.add_argument("--bed", default=None)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    pipe = CareVoicePipeline(model_dir=args.model_dir, device=args.device)
    print(f"[pipeline] models loaded in {pipe.load_ms:.0f} ms")
    print(json.dumps(pipe.process(args.audio_path, bed=args.bed), indent=2))
