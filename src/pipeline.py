"""
CareVoicePipeline: the one entry point (CLAUDE.md section 2.7).

The Streamlit app and the end-to-end evaluation both import this class. Never write a
second implementation of the pipeline anywhere else.

    audio -> load_audio (-23 LUFS) -> energy pre-filter
          -> load_audio_acoustic (RMS 0.05, crop 4 s, no pad) -> WavLM Base+ (frozen,
             mean-pooled over real frames) -> arousal head (fear, non_speech)
          -> load_audio_native (no resampling) -> HNR -> z vs this bed's baseline (prosodic distress)
          -> denoise (DISABLED, see preprocess.CONFIG) -> trim -> Whisper small (faster-whisper int8)
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

import gc
import hashlib
import json
import logging
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

from src.mapping import care_request_tag, class_base_priority, medication_tag
from src.preprocess import SPEECH_GUARD
from src.preprocess import CONFIG as PREPROCESS_CONFIG
from src import prosody
from src.baseline import BedBaseline
from src.preprocess import DENOISE_DISABLED_REASON, AudioLoadError, load_audio_native
from src.preprocess import (ACOUSTIC_CONTRACT, denoise, energy_present, load_audio,
                            load_audio_acoustic, trim)
from src.rules import CONFIG as RULES_CONFIG
from src.rules import UNCLASSIFIED, apply_rules, speech_guard

log = logging.getLogger(__name__)

# Every model file the pipeline cannot run without, checked before anything loads.
# The WavLM distress gate is retired (never fired on real voices); distress is now
# prosodic (src/prosody.py) and needs no model file.
REQUIRED_MODELS = ("whisper_dir", "bert_dir", "wavlm_dir", "arousal_head")

CONFIG = {
    "sr": 16000,
    "seed": 42,

    # Subfolders / files of model_dir. Whisper and WavLM come from scripts/fetch_models.py.
    "whisper_dir": "faster-whisper-small",
    "wavlm_dir": "wavlm-base-plus",
    "bert_dir": "clinicalbert_requestclass",
    # Pretrained text emotion model (scripts/fetch_models.py). The methodology
    # diagram has the text branch producing request type AND sentiment; ClinicalBERT
    # was fine-tuned for request type only, so sentiment comes from this model.
    "emotion_dir": "emotion-english-distilroberta-base",
    "arousal_head": "arousal_head.joblib",

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
    def __init__(self, model_dir="models", device="cpu", load_emotion=False, baseline_db=None):
        """load_emotion: the text-emotion model is display-only (it never affects
        priority) and costs 313 MB of weights, so it is off by default and not
        loaded at all. set_emotion() turns it on or off in a running pipeline."""
        t0 = time.perf_counter()
        self.model_dir = Path(model_dir)
        self.device = device
        self.emo_model = self.emo_tokenizer = self.emo_labels = None
        # Per-bed prosodic baseline (SQLite). Without one - tests, batch evaluation -
        # every bed is permanently in cold start: no prosodic score, no escalation.
        self.baseline = BedBaseline(baseline_db) if baseline_db else None

        random.seed(CONFIG["seed"])
        np.random.seed(CONFIG["seed"])
        torch.manual_seed(CONFIG["seed"])

        # Refuse to start, at once and naming every missing file, rather than
        # loading 1 GB of other models first or running without the distress gate.
        # A nurse-call system missing a model must fail at startup, loudly, never
        # mid-shift on a call.
        required = REQUIRED_MODELS + (("emotion_dir",) if load_emotion else ())
        missing = [str(self.model_dir / CONFIG[k]) for k in required
                   if not (self.model_dir / CONFIG[k]).exists()]
        if missing:
            raise FileNotFoundError(
                "CareVoicePipeline cannot start, missing model files: " + ", ".join(missing)
                + ". Run `python scripts/fetch_models.py` for Whisper/WavLM/emotion; the "
                  "trained heads and ClinicalBERT come from the Kaggle jobs.")

        # Say so once, at startup, rather than let a disabled stage look active.
        if not PREPROCESS_CONFIG["enable_denoise"]:
            log.warning("denoising disabled: %s", DENOISE_DISABLED_REASON)

        self._load_asr()
        self._load_text()
        if load_emotion:
            self._load_emotion()
        self._load_acoustic()
        if CONFIG["warmup"]:
            self._warmup()
        self.model_versions = self._model_versions()
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

    def _load_emotion(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        emo_dir = self._path("emotion_dir")
        self.emo_tokenizer = AutoTokenizer.from_pretrained(emo_dir, local_files_only=True)
        self.emo_model = AutoModelForSequenceClassification.from_pretrained(
            emo_dir, local_files_only=True).to(self.device).eval()
        self.emo_labels = [self.emo_model.config.id2label[i]
                           for i in range(self.emo_model.config.num_labels)]
        if "anger" not in self.emo_labels:
            raise ValueError(f"emotion model has no 'anger' label: {self.emo_labels}")

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

        # Refuse a head fitted on differently prepared features: its scores would be
        # meaningless and nothing downstream could tell.
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

        # rules.py is the single source of truth for thresholds. The copy saved with
        # the head is only checked, never used.
        for name, saved, key in (("arousal_head", arousal.get("non_speech_threshold"), "NON_SPEECH_HIGH"),):
            if saved is not None and abs(saved - RULES_CONFIG[key]) > 1e-9:
                warnings.warn(f"{name}.joblib threshold {saved} differs from "
                              f"rules.CONFIG[{key!r}]={RULES_CONFIG[key]}; using rules.py")

    def _warmup(self):
        rng = np.random.default_rng(CONFIG["seed"])
        wav = (0.05 * rng.standard_normal(CONFIG["sr"])).astype(np.float32)
        self._acoustic(wav)
        self._transcribe(trim(wav, sr=CONFIG["sr"]))
        self._classify("warm up")
        prosody.hnr_db(np.tile(wav, 3), CONFIG["sr"])          # loads openSMILE once
        if self.emo_model is not None:
            self._emotion("warm up")

    # ------------------------------------------------------------------ runtime switches

    @property
    def emotion_enabled(self) -> bool:
        return self.emo_model is not None

    def set_emotion(self, enabled: bool):
        """Load or free the display-only emotion model in place, without reloading
        anything else (a second pipeline would double the resident weights)."""
        if enabled and self.emo_model is None:
            self._load_emotion()
            self._emotion("warm up")
        elif not enabled and self.emo_model is not None:
            self.emo_model = self.emo_tokenizer = self.emo_labels = None
            gc.collect()
        self.model_versions = self._model_versions()

    def keep_warm(self) -> float:
        """One dummy inference through every loaded model; returns ms. Touching the
        weights keeps the OS from swapping them out between calls. It only helps
        when the machine has free RAM: under memory pressure it just pushes other
        processes (the browser) out instead. Called by the app's keep-warm switch."""
        t0 = time.perf_counter()
        self._warmup()
        return _ms(t0)

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

    @torch.no_grad()
    def _emotion(self, transcript: str) -> dict[str, float]:
        enc = self.emo_tokenizer(transcript, return_tensors="pt", truncation=True,
                                 max_length=CONFIG["bert_max_length"]).to(self.device)
        probs = torch.softmax(self.emo_model(**enc).logits, dim=-1)[0].cpu().numpy()
        return {c: float(p) for c, p in zip(self.emo_labels, probs)}

    def _model_versions(self) -> dict:
        """Short content hashes of the trained/downloaded weights, logged with every
        decision so it can be tied to the exact models that made it."""
        out = {}
        for key in ("arousal_head",):
            out[key] = hashlib.sha256(self._path(key).read_bytes()).hexdigest()[:12]
        f = self._path("bert_dir") / "model.safetensors"
        out["bert_dir"] = hashlib.sha256(f.read_bytes()).hexdigest()[:12] if f.exists() else None
        out["emotion_dir"] = None
        if self.emo_model is not None:
            f = self._path("emotion_dir") / "config.json"
            out["emotion_dir"] = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
        return out

    def _acoustic(self, wav: np.ndarray) -> dict:
        """The arousal head on one unpadded embedding: fear (display only) and
        non_speech (the speech guard). Distress is prosodic, see process()."""
        emb = embed_clip(self.feature_extractor, self.wavlm, wav, self.device)
        ap = self.arousal_model.predict_proba(emb)[0]
        return {
            "arousal": float(ap[self.fear_index]),
            "non_speech_prob": float(ap[self.non_speech_index]),
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

        # A missing, empty or corrupt recording must still produce a routed call:
        # someone pressed the button. The error becomes the decision's reason.
        audio_error = None
        t0 = time.perf_counter()
        try:
            wav, meta = load_audio(audio_path, target_sr=CONFIG["sr"], return_meta=True)
            energy_ok, energy_reason = energy_present(meta)
        except AudioLoadError as e:
            audio_error = str(e)
            log.warning("bed %s: %s; routing at P%d", bed, audio_error, RULES_CONFIG["P_NO_SPEECH"])
            wav, energy_ok, energy_reason = np.zeros(0, dtype=np.float32), False, ""
        lat["load"] = _ms(t0)
        duration_s = len(wav) / CONFIG["sr"]

        # --- acoustic branch ---
        acoustic = {"arousal": None, "non_speech_prob": None}
        lat["load_acoustic"] = lat["wavlm_and_heads"] = 0.0
        if energy_ok:
            t0 = time.perf_counter()
            try:
                wav_ac = load_audio_acoustic(audio_path, target_sr=CONFIG["sr"])
            except Exception as e:  # readable by soundfile above but not here: report it
                audio_error = f"acoustic loader failed ({type(e).__name__}: {e})"
                log.warning("bed %s: %s; routing at P%d", bed, audio_error,
                            RULES_CONFIG["P_NO_SPEECH"])
                energy_ok = False
            lat["load_acoustic"] = _ms(t0)
            if audio_error is None:
                t0 = time.perf_counter()
                acoustic = self._acoustic(wav_ac)
                lat["wavlm_and_heads"] = _ms(t0)

        # --- prosodic distress: HNR at the native rate vs this bed's baseline ---
        hnr = native_sr = z = None
        stats = None
        lat["prosody"] = 0.0
        if energy_ok:
            t0 = time.perf_counter()
            try:
                wav_native, native_sr = load_audio_native(audio_path)
                hnr = prosody.hnr_db(wav_native, native_sr)
            except Exception as e:  # a failed feature must never block the call
                log.warning("bed %s: prosody failed (%s: %s); no distress score", bed,
                            type(e).__name__, e)
            if self.baseline is not None and bed is not None and native_sr:
                stats = self.baseline.stats(bed, native_sr)
                if stats.ready:
                    z = prosody.distress_z(hnr, stats.median, stats.std)
            lat["prosody"] = _ms(t0)
        distress = prosody.distress_score(z, RULES_CONFIG["DISTRESS_SCORE"])

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
            distress, energy_reason)
        if audio_error:
            speech_reason = f"audio could not be read ({audio_error})"
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

        # Sentiment (text emotion) and the medication keyword tag, on the same
        # transcript. DISPLAY ONLY: neither changes the class or the priority.
        # Text anger was tried as a distress-suppression signal and removed:
        # "I feel cold" scores anger 0.97, a plainly angry sentence 0.67, and 27 of
        # 706 genuine requests score above 0.5 (see rules.py).
        t0 = time.perf_counter()
        sentiment = (self._emotion(transcript)
                     if transcript.strip() and self.emo_model is not None else {})
        sentiment_label = max(sentiment, key=sentiment.get) if sentiment else None
        text_anger = sentiment.get("anger")
        medication = medication_tag(transcript)
        care_request = care_request_tag(transcript)
        lat["sentiment"] = _ms(t0)

        # --- the rule ---
        t0 = time.perf_counter()
        decision = apply_rules(
            base_priority=base_priority,
            request_class=request_class,
            class_confidence=class_confidence,
            distress=distress,
            asr_confidence=asr_confidence,
            energy_ok=energy_ok,
            non_speech_prob=acoustic["non_speech_prob"],
            energy_reason=energy_reason,
            transcript=transcript,
            asr_heard=heard,
            audio_error=audio_error,
            care_request=care_request,
        )
        lat["rules"] = _ms(t0)

        # A call joins its bed's baseline only if it is an ordinary call: speech,
        # a measurable HNR, not escalated, not Emergency. An alarming call must
        # never become the new normal for that bed.
        baseline_joined = bool(
            self.baseline is not None and bed is not None and hnr is not None
            and has_speech and not decision.escalated and request_class != "Emergency")
        if baseline_joined:
            self.baseline.add(bed, native_sr, hnr)
        lat["total"] = _ms(t_all)

        return {
            "audio_path": str(audio_path),
            "bed": bed,
            "duration_s": round(duration_s, 3),
            "audio_error": audio_error,
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
            "medication": medication,
            "care_request": care_request,
            "sentiment": sentiment,
            "sentiment_label": sentiment_label,
            "sentiment_enabled": self.emo_model is not None,
            "text_anger": text_anger,
            "base_priority": base_priority,
            "arousal": acoustic["arousal"],
            "hnr": hnr,
            "prosody_sr": native_sr,
            "baseline_n": stats.n if stats else 0,
            "baseline_min": RULES_CONFIG["BASELINE_MIN_CALLS"],
            "baseline_ready": bool(stats and stats.ready),
            "baseline_joined": baseline_joined,
            "distress_z": z,
            "distress": distress,
            "distress_fired": decision.distress_fired,
            "final_priority": decision.final_priority,
            "escalated": decision.escalated,
            "rules_fired": decision.rules_fired,
            "reason": decision.reason,
            # Everything needed to reconstruct this decision after the config moves.
            "thresholds": {**{k: v for k, v in RULES_CONFIG.items() if k != "STARVATION_SECONDS"},
                           "MIN_RAW_RMS": SPEECH_GUARD["MIN_RAW_RMS"],
                           "denoise_enabled": PREPROCESS_CONFIG["enable_denoise"],
                           "acoustic_contract": ACOUSTIC_CONTRACT},
            "model_versions": self.model_versions,
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
