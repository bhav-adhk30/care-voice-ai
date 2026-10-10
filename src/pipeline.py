"""
CareVoicePipeline: the one entry point (CLAUDE.md section 2.7).

The Streamlit app and the end-to-end evaluation both import this class. Never write a
second implementation of the pipeline anywhere else.

    audio -> load_audio -> denoise -> trim
                                        |-> Whisper small (faster-whisper int8) -> ClinicalBERT -> request class
                                        |-> WavLM Base+ (frozen, mean-pooled) -> arousal head + distress gate
          -> apply_rules (src/rules.py) -> final priority + reason

The two branches never share anything but the cleaned waveform. They meet only in
apply_rules, which is a deterministic rule and not a trained model (section 2.1).

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
from src.preprocess import denoise, load_audio, pad_or_crop, speech_present, trim
from src.rules import CONFIG as RULES_CONFIG
from src.rules import UNCLASSIFIED, apply_rules

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
        # is handled by preprocess.speech_present instead.
        "vad_filter": False,
    },
    # No speech decoded at all: treat the transcript as maximally untrusted. Any value
    # below rules.CONFIG["ASR_MIN"] works; this one is JSON-safe, unlike -inf.
    "empty_transcript_logprob": -10.0,

    # Text classifier.
    "bert_max_length": 128,

    # The acoustic heads were trained on fixed 4 s windows (cropped from the start,
    # zero-padded at the end) - see notebooks/04_acoustic_branch.ipynb. Inference must
    # feed WavLM the same window, or the mean pool averages over a different amount of
    # padding and the measured thresholds no longer hold.
    "acoustic_window_s": 4.0,

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
def embed(feature_extractor, wavlm, wav: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Frozen WavLM Base+, last hidden state mean-pooled over time -> (1, 768).

    Fixed window, cropped from the start and zero-padded at the end - the same
    window notebook 04 used to train both heads. Note that training then added
    Gaussian noise over the whole window, padding included (degrade()), so the
    heads saw noise-filled padding, never digital zeros. notebooks/09 measures
    what that does to the scores."""
    win = pad_or_crop(wav, int(round(CONFIG["acoustic_window_s"] * CONFIG["sr"])))
    inp = feature_extractor(win.astype(np.float32), sampling_rate=CONFIG["sr"],
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
        self.arousal_model = arousal["model"]
        self.fear_index = int(arousal["fear_index"])
        if arousal["classes"][self.fear_index] != "fear":
            raise ValueError(f"arousal head fear_index={self.fear_index} points at "
                             f"{arousal['classes'][self.fear_index]!r}, not 'fear'")

        distress = joblib.load(self._path("distress_gate"))
        self.distress_model = distress["model"]
        self.distress_pos_index = list(self.distress_model.classes_).index(1)

        # rules.py is the single source of truth for thresholds. The copies saved with
        # the heads are only checked, never used.
        for name, saved, key in (("arousal_head", arousal.get("threshold"), "AROUSAL_HIGH"),
                                 ("distress_gate", distress.get("threshold"), "DISTRESS_HIGH")):
            if saved is not None and abs(saved - RULES_CONFIG[key]) > 1e-9:
                warnings.warn(f"{name}.joblib threshold {saved} differs from "
                              f"rules.CONFIG[{key!r}]={RULES_CONFIG[key]}; using rules.py")

    def _warmup(self):
        rng = np.random.default_rng(CONFIG["seed"])
        wav = (0.05 * rng.standard_normal(CONFIG["sr"])).astype(np.float32)
        wav = trim(wav, sr=CONFIG["sr"])
        self._transcribe(wav)
        self._classify("warm up")
        self._embed(wav)

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

    def _embed(self, wav: np.ndarray) -> np.ndarray:
        return embed(self.feature_extractor, self.wavlm, wav, self.device)

    # ------------------------------------------------------------------ entry point

    def process(self, audio_path, bed=None) -> dict:
        """
        Run one clip through the whole system. Returns every intermediate, not just the
        final priority, so the dashboard can show WHY a call was escalated.
        """
        lat = {}
        t_all = time.perf_counter()

        t0 = time.perf_counter()
        wav, meta = load_audio(audio_path, target_sr=CONFIG["sr"], return_meta=True)
        lat["load"] = _ms(t0)
        duration_s = len(wav) / CONFIG["sr"]

        t0 = time.perf_counter()
        wav = denoise(wav, sr=CONFIG["sr"])
        lat["denoise"] = _ms(t0)

        t0 = time.perf_counter()
        wav = trim(wav, sr=CONFIG["sr"])
        lat["trim"] = _ms(t0)

        # Energy and duration only, never the transcript: a wordless groan has no
        # transcript and must still reach the distress gate.
        has_speech, speech_reason = speech_present(meta, len(wav), sr=CONFIG["sr"])

        # --- text branch ---
        # Skipped entirely when there is no speech. ClinicalBERT on an empty string
        # is not a fallback, it is the same out-of-distribution problem as the
        # distress gate on silence: it predicted Emergency on a silent clip.
        # UNCLASSIFIED is a pipeline sentinel, not a trained class; it is not in
        # mapping.CLASSES and the dashboard renders it as a grey badge.
        if has_speech:
            t0 = time.perf_counter()
            transcript, asr_confidence = self._transcribe(wav)
            lat["asr"] = _ms(t0)

            t0 = time.perf_counter()
            class_probs = self._classify(transcript)
            request_class = max(class_probs, key=class_probs.get)
            class_confidence = class_probs[request_class]
            base_priority = class_base_priority(request_class)
            lat["text_classifier"] = _ms(t0)
        else:
            transcript, asr_confidence = "", None
            request_class, class_probs, class_confidence = UNCLASSIFIED, {}, 0.0
            base_priority = RULES_CONFIG["P_NO_SPEECH"]
            lat["asr"] = lat["text_classifier"] = 0.0

        # --- acoustic branch ---
        # Skipped entirely when there is no speech. The distress gate scores a WavLM
        # embedding of pure silence at 0.9996: silence lies outside both of its
        # training classes and the logistic regression extrapolates confidently in
        # an arbitrary direction. The dashboard shows 0.0 rather than that number.
        if has_speech:
            t0 = time.perf_counter()
            emb = self._embed(wav)
            lat["wavlm"] = _ms(t0)

            t0 = time.perf_counter()
            arousal = float(self.arousal_model.predict_proba(emb)[0, self.fear_index])
            distress = float(self.distress_model.predict_proba(emb)[0, self.distress_pos_index])
            lat["acoustic_heads"] = _ms(t0)
        else:
            arousal, distress = 0.0, 0.0
            lat["wavlm"] = lat["acoustic_heads"] = 0.0

        # --- the rule ---
        t0 = time.perf_counter()
        decision = apply_rules(
            base_priority=base_priority,
            request_class=request_class,
            class_confidence=class_confidence,
            arousal=arousal,
            distress=distress,
            asr_confidence=asr_confidence,
            speech_present=has_speech,
            speech_reason=speech_reason,
        )
        lat["rules"] = _ms(t0)
        lat["total"] = _ms(t_all)

        return {
            "audio_path": str(audio_path),
            "bed": bed,
            "duration_s": round(duration_s, 3),
            "transcript": transcript,
            "asr_confidence": asr_confidence,
            "asr_trusted": decision.asr_trusted,
            "speech_present": has_speech,
            "speech_reason": speech_reason,
            "request_class": request_class,
            "class_probs": class_probs,
            "class_confidence": class_confidence,
            "base_priority": base_priority,
            "arousal": arousal,
            "distress": distress,
            "arousal_fired": decision.arousal_fired,
            "distress_fired": decision.distress_fired,
            "final_priority": decision.final_priority,
            "escalated": decision.escalated,
            "reason": decision.reason,
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
