"""
Care-Voice AI demo: nurse station, call submission, and results.

    HF_HUB_OFFLINE=1 streamlit run app/streamlit_app.py

Imports CareVoicePipeline from src/pipeline.py, the same class the end-to-end
evaluation uses (CLAUDE.md section 2.7). There is no second implementation of the
pipeline here: this file only captures audio, stores, orders and displays what
process() returns. Audio of any rate or channel count goes to process() as-is;
its two loaders (src/preprocess.py) resample and downmix, so no preprocessing
happens in the app (section 2.5).

Storage, both SQLite, both gitignored:
  app/calls.db       the live queue (waiting / answered). Mutable operational state.
  app/audit_log.db   src/audit.py: one append-only row per decision, including the
                     thresholds in force. Nothing in the app can update or delete it.

Cached mode replays app/cached_results.json (scripts/build_demo_cache.py) so the
demo still runs if a model fails to load on the day.
"""

from __future__ import annotations

import heapq
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never let a network call hang the demo
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.audit import AuditLog
from src.mapping import CLASS_BASE_PRIORITY, MEDICATION_CLASS
from src.rules import CONFIG as RULES
from src.rules import EMERGENCY_BUTTON, UNCLASSIFIED, starvation_priority

CONFIG = {
    "model_dir": ROOT / "models",
    "demo_dir": ROOT / "app" / "demo_clips",
    "cache_file": ROOT / "app" / "cached_results.json",
    "queue_db": ROOT / "app" / "calls.db",
    "audit_db": ROOT / "app" / "audit_log.db",
    "results_dir": ROOT / "results",
    "queue_refresh_s": 1,          # the waiting timer ticks every second
    # 48 kHz, not 16 kHz: the prosodic distress feature (HNR) is computed at the
    # native rate and loses 4-8 dB - and for some speakers the whole calm/distress
    # difference - when audio is captured or resampled at 16 kHz. Whisper and WavLM
    # get their own 16 kHz copies inside process().
    "mic_sample_rate": 48000,
    # Everything phones, QuickTime and browsers produce. All decode through
    # src/preprocess.py (libsndfile, else ffmpeg); a video's audio track is used.
    "audio_types": ["wav", "mp3", "m4a", "mp4", "mov", "qta", "aac", "flac", "ogg", "opus",
                    "webm", "aiff", "aif", "caf", "3gp", "amr"],
    "beds": [str(b) for b in range(1, 31)],
    "demo_clock": [1, 10, 60],     # waiting time multiplier, so starvation is visible on stage
    "keep_warm_s": 30,
    "emergency_debounce_s": 3,     # a second press for the same bed this soon is a double-tap             # dummy-inference interval when the keep-warm switch is on
}

st.set_page_config(page_title="Care-Voice AI", page_icon="🔔", layout="wide")


# --------------------------------------------------------------------------- look and feel

# Presentation only. Light theme (.streamlit/config.toml) because projectors wash out
# dark ones; colours are defined once here.
CSS = """
<style>
:root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --card:#ffffff; }
.block-container { padding-top: 1.1rem; max-width: 1440px; }
header[data-testid="stHeader"] { background: transparent; }
.cv-hero { display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding:18px 24px; border-radius:16px; margin-bottom:14px;
  background:linear-gradient(120deg,#0b1f3a 0%,#123a6e 60%,#1d4ed8 100%); color:#fff; }
.cv-hero h1 { font-size:1.65rem; margin:0; color:#fff; letter-spacing:-0.01em; }
.cv-hero p { margin:3px 0 0; color:#c9d7ee; font-size:0.95rem; }
.cv-pill { display:inline-block; padding:5px 12px; border-radius:999px; font-size:0.78rem;
  font-weight:600; background:rgba(255,255,255,0.14); color:#fff; margin-left:6px; white-space:nowrap; }
.cv-pill.ok::before { content:"●"; color:#4ade80; margin-right:6px; }
.cv-pill.warn::before { content:"●"; color:#fbbf24; margin-right:6px; }
.cv-kpi { border:1px solid var(--line); border-radius:14px; padding:12px 18px; background:var(--card); }
.cv-kpi .v { font-size:1.9rem; font-weight:750; color:var(--ink); line-height:1.15; }
.cv-kpi .l { font-size:0.74rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; }
.cv-tile { border-radius:12px; padding:10px 4px; text-align:center; font-weight:750; }
.cv-tile .p { font-size:1.55rem; line-height:1.05; }
.cv-tile .t { font-size:0.62rem; letter-spacing:0.09em; text-transform:uppercase; opacity:0.92; }
.cv-tile.big { padding:16px 6px; } .cv-tile.big .p { font-size:2.4rem; } .cv-tile.big .t { font-size:0.75rem; }
.cv-bed { font-size:1.35rem; font-weight:750; color:var(--ink); line-height:1.1; }
.cv-sub { font-size:0.78rem; color:var(--muted); }
.cv-quote { font-size:1.06rem; color:var(--ink); margin:4px 0 2px; }
.cv-why { font-size:0.86rem; color:#334155; }
.cv-chip { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:650;
  font-size:0.8rem; white-space:nowrap; margin:0 4px 4px 0; }
.cv-step { display:inline-block; padding:10px 14px; border-radius:12px; background:#fff;
  border:1px solid var(--line); font-size:0.9rem; color:var(--ink); }
.cv-step b { display:block; font-size:0.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; }
.cv-arrow { display:inline-block; margin:0 8px; color:#94a3b8; font-size:1.2rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:var(--card); border-radius:14px; }
.st-key-emergency_btn button { background:#dc2626; border:0; color:#fff; font-size:1.25rem;
  font-weight:800; padding:18px 10px; border-radius:14px; letter-spacing:0.02em;
  box-shadow:0 6px 18px rgba(220,38,38,0.35); }
.st-key-emergency_btn button:hover { background:#b91c1c; color:#fff; }
.st-key-beep { display:none; }
@keyframes cvflash { 0%,100% { background:#dc2626; } 50% { background:#7f1d1d; } }
.cv-alarm { animation:cvflash 1s infinite; color:#fff; border-radius:14px; padding:14px 20px;
  font-size:1.15rem; font-weight:750; margin:6px 0 10px; }
</style>
"""

PRIORITY_STYLE = {          # background, text, label
    1: ("#b91c1c", "#ffffff", "Immediate"),
    2: ("#ea580c", "#ffffff", "Urgent"),
    3: ("#eab308", "#1f2937", "Standard"),
    4: ("#16a34a", "#ffffff", "Routine"),
}
CLASS_STYLE = {
    "Emergency": ("#fee2e2", "#991b1b", "🚨 Emergency"),
    "Pain": ("#ffedd5", "#9a3412", "🩹 Pain"),
    "Assistance": ("#dbeafe", "#1e40af", "🤝 Assistance"),
    "General": ("#e2e8f0", "#334155", "💬 General"),
    MEDICATION_CLASS: ("#ede9fe", "#5b21b6", "💊 Medication"),
    UNCLASSIFIED: ("#f1f5f9", "#475569", "🔔 Call button (no words)"),
    EMERGENCY_BUTTON: ("#dc2626", "#ffffff", "🚨 Emergency button"),
}

# Plain-language versions of the rule ids in rules.Decision.rules_fired.
RULE_WORDS = {
    "R0_no_speech": "No words detected: treated like a call-button press",
    "R0_audio_error": "Recording unreadable: treated like a call-button press",
    "R1_distress_untrusted_transcript": "Sounds distressed and the words are unclear",
    "R2_distress": "Voice sounds distressed for this patient",
    "R4_emergency_floor": "Emergency request: always kept urgent",
    "R5_care_request": "Needs help to the toilet or to move",
    "R6_distress_words": "Urgent words (\"please come\", \"oh no!\")",
    "R7_comfort_request": "Comfort request (window, light, TV...)",
    "R8_medication": "Medication request",
    "R9_emergency_button": "Patient pressed the EMERGENCY button: go now",
}

SENTIMENT_EMOJI = {"anger": "😠", "disgust": "😖", "fear": "😨", "joy": "🙂", "neutral": "😐",
                   "sadness": "😢", "surprise": "😮"}

# Demo clips are named NN_what__speaker.wav; each speaker keeps one bed, because the
# per-bed voice baseline assumes one patient per bed (as on a real ward).
SPEAKER_BED = {"bhavishya": "7", "sumayah": "12", "vinuthna": "3"}


def _chip(text: str, bg: str, fg: str) -> str:
    return f"<span class='cv-chip' style='background:{bg};color:{fg}'>{text}</span>"


def priority_tile(p: int, big: bool = False) -> str:
    bg, fg, label = PRIORITY_STYLE[p]
    return (f"<div class='cv-tile{' big' if big else ''}' style='background:{bg};color:{fg}'>"
            f"<div class='p'>P{p}</div><div class='t'>{label}</div></div>")


def priority_badge(p: int, short: bool = False) -> str:
    bg, fg, label = PRIORITY_STYLE[p]
    return _chip(f"P{p}" if short else f"P{p} {label.lower()}", bg, fg)


def class_badge(request_class: str, medication: list | None = None,
                care: list | None = None, words: list | None = None) -> str:
    bg, fg, label = CLASS_STYLE.get(request_class, CLASS_STYLE["General"])
    html = _chip(label, bg, fg)
    if medication and request_class != MEDICATION_CLASS:
        html += _chip("💊 mentions medication", "#f5f3ff", "#5b21b6")
    if care:
        html += _chip("🚻 toilet / mobility", "#ccfbf1", "#115e59")
    if words:
        html += _chip("❗ urgent words", "#fce7f3", "#9d174d")
    return html


def sentiment_chip(r: dict) -> str:
    """Sentiment of the words (display only, never priority)."""
    lab = r.get("sentiment_label")
    if not lab:
        return ""
    p = (r.get("sentiment") or {}).get(lab)
    return (f"<span class='cv-sub'>Sentiment: {SENTIMENT_EMOJI.get(lab, '')} {lab}"
            f"{f' ({p:.0%})' if p is not None else ''}</span>")


def fmt_wait(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def plain_reason(r: dict, bumps: list[int] | None = None, waited: float = 0.0) -> str:
    """What the nurse reads: the rules that fired, in words, plus starvation."""
    lines = [RULE_WORDS.get(f, f) for f in r.get("rules_fired", [])]
    if not lines:
        lines = (["Words unclear: check in person"] if not r["asr_trusted"]
                 else ["Priority from what they asked for"])
    if bumps:
        lines.append(f"⏱️ Waited {int(waited // 60)} min: raised from P{r['final_priority']}")
    return " · ".join(lines)


def quote(r: dict) -> str:
    t = (r.get("transcript") or "").strip()
    if r.get("request_class") == EMERGENCY_BUTTON:
        return "<i>Emergency button pressed</i>"
    return f"“{t}”" if t else "<i>(no words)</i>"


@st.cache_data
def beep_wav() -> bytes:
    """Three short 880 Hz pips: the nurse-station alert for a new P1 call."""
    import io

    import numpy as np
    import soundfile as sf
    sr = 22050
    pip = np.sin(2 * np.pi * 880 * np.arange(int(0.16 * sr)) / sr) * np.hanning(int(0.16 * sr))
    gap = np.zeros(int(0.09 * sr))
    wav = np.concatenate([pip, gap, pip, gap, pip]).astype(np.float32) * 0.6
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


def kpi(col, value, label):
    col.markdown(f"<div class='cv-kpi'><div class='v'>{value}</div><div class='l'>{label}</div></div>",
                 unsafe_allow_html=True)


# --------------------------------------------------------------------------- storage

@st.cache_resource
def queue_db() -> sqlite3.Connection:
    con = sqlite3.connect(CONFIG["queue_db"], check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS queue (
        id INTEGER PRIMARY KEY, audit_id INTEGER, bed TEXT, submitted_at REAL,
        result TEXT, status TEXT DEFAULT 'waiting', answered_at REAL)""")
    con.commit()
    return con


@st.cache_resource
def audit_log() -> AuditLog:
    return AuditLog(CONFIG["audit_db"])


def enqueue(bed: str, audit_id: int, result: dict):
    queue_db().execute("INSERT INTO queue (audit_id, bed, submitted_at, result) VALUES (?,?,?,?)",
                       (audit_id, bed, time.time(), json.dumps(result)))
    queue_db().commit()


def answer(qid: int):
    queue_db().execute("UPDATE queue SET status='answered', answered_at=? WHERE id=?", (time.time(), qid))
    queue_db().commit()


def waiting_queue(clock: float) -> list[dict]:
    """Waiting calls, re-prioritised by the starvation timer on every refresh,
    ordered by effective priority then by how long they have waited."""
    now, heap = time.time(), []
    for qid, bed, ts, res in queue_db().execute(
            "SELECT id, bed, submitted_at, result FROM queue WHERE status='waiting'"):
        r = json.loads(res)
        waited = (now - ts) * clock
        eff, bumps = starvation_priority(r["final_priority"], waited)
        heapq.heappush(heap, (eff, ts, qid, {"qid": qid, "bed": bed, "waited": waited,
                                             "eff": eff, "bumps": bumps, **r}))
    return [heapq.heappop(heap)[3] for _ in range(len(heap))]


# --------------------------------------------------------------------------- pipeline

@st.cache_resource(show_spinner="Loading models (Whisper, ClinicalBERT, emotion, WavLM)...")
def load_pipeline():
    from src.pipeline import CareVoicePipeline
    # Emotion model never loaded here: it follows the sidebar toggle via set_emotion().
    # The per-bed prosodic baseline lives next to the queue in app/calls.db.
    pipe = CareVoicePipeline(model_dir=str(CONFIG["model_dir"]), device="cpu", load_emotion=False,
                             baseline_db=str(CONFIG["queue_db"]))
    st.session_state["pipeline_loaded"] = True
    return pipe


def apply_emotion_toggle():
    """Load or free the display-only emotion model in the running pipeline. Does
    nothing before the pipeline exists: it is applied at the first call instead."""
    if st.session_state.get("pipeline_loaded"):
        load_pipeline().set_emotion(st.session_state.get("emotion_on", True))


def load_cache() -> dict:
    try:
        return json.loads(CONFIG["cache_file"].read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def press_emergency(bed: str) -> dict:
    """The bedside emergency button: no audio, no model, works in cached mode too.
    src.pipeline.emergency_button_result builds the call; it is logged and queued
    like any other."""
    from src.pipeline import emergency_button_result
    r = {**emergency_button_result(bed), "cached": False}
    audit_id = audit_log().append(r, bed=bed, audio_name="emergency_button", audio_bytes=b"",
                                  mode="button")
    enqueue(bed, audit_id, r)
    return {**r, "audit_id": audit_id}


def on_emergency_click():
    """Runs exactly once per click. A second press for the same bed within
    CONFIG['emergency_debounce_s'] is a double-tap, not a new emergency."""
    bed, now = st.session_state.get("eb_bed", "14"), time.time()
    last = st.session_state.get("eb_last", {})
    if now - last.get(bed, 0) < CONFIG["emergency_debounce_s"]:
        return
    st.session_state["eb_last"] = {**last, bed: now}
    st.session_state["last_result"] = press_emergency(bed)


def run_call(name: str, data: bytes, bed: str, cached_mode: bool) -> dict | None:
    if cached_mode:
        r = load_cache().get(name)
        if r is None:
            st.error(f"No cached result for {name!r}. Cached mode only replays clips in "
                     f"{CONFIG['cache_file'].name}; run scripts/build_demo_cache.py.")
            return None
        r = {**r, "bed": bed, "cached": True}
    else:
        pipe = load_pipeline()
        pipe.set_emotion(st.session_state.get("emotion_on", True))
        with tempfile.NamedTemporaryFile(suffix=Path(name).suffix or ".wav", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            r = pipe.process(tmp, bed=bed)
        finally:
            os.unlink(tmp)
        r = {**r, "audio_path": name, "cached": False}
    audit_id = audit_log().append(r, bed=bed, audio_name=name, audio_bytes=data,
                                  mode="cached" if cached_mode else "live")
    enqueue(bed, audit_id, r)
    return {**r, "audit_id": audit_id}


# --------------------------------------------------------------------------- header

def header(cached_mode: bool):
    models = ("Cached mode" if cached_mode else
              "Models loaded" if st.session_state.get("pipeline_loaded") else "Models load on first call")
    st.markdown(
        f"<div class='cv-hero'><div><h1>🔔 Care-Voice AI</h1>"
        f"<p>Patients speak instead of pressing a button. The nurse station sees what they need "
        f"and how urgently.</p></div><div>"
        f"<span class='cv-pill {'warn' if cached_mode else 'ok'}'>{models}</span>"
        f"<span class='cv-pill'>Offline · on-device</span></div></div>",
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- tab 1

def alarm(q: list[dict]):
    """Flashing banner for every waiting P1 call, and a beep when a new one arrives
    (emergency button, a voice emergency, or a call the waiting timer pushed to P1).
    Runs inside the 1 s queue fragment, so it alerts whichever tab is open."""
    p1 = [c for c in q if c["eff"] == 1]
    seen = st.session_state.setdefault("alerted", set())
    new = [c["qid"] for c in p1 if c["qid"] not in seen]
    if new:
        seen.update(new)
        st.session_state["beep_until"] = time.time() + 2.5
    if p1:
        beds = ", ".join(f"Bed {c['bed']}" + (" (button)" if c["request_class"] == EMERGENCY_BUTTON else "")
                         for c in p1)
        st.markdown(f"<div class='cv-alarm'>🚨 IMMEDIATE — {beds}</div>", unsafe_allow_html=True)
    if st.session_state.get("alert_sound", True) and time.time() < st.session_state.get("beep_until", 0):
        with st.container(key="beep"):
            st.audio(beep_wav(), format="audio/wav", autoplay=True)


def tab_nurse_station():

    @st.fragment(run_every=CONFIG["queue_refresh_s"])
    def queue_view():
        clock = st.session_state.get("demo_clock", 1)
        q = waiting_queue(clock)
        alarm(q)
        k = st.columns(4)
        kpi(k[0], len(q), "Calls waiting")
        kpi(k[1], sum(c["eff"] == 1 for c in q), "P1 immediate")
        kpi(k[2], sum(c["eff"] == 2 for c in q), "P2 urgent")
        kpi(k[3], fmt_wait(max((c["waited"] for c in q), default=0)), "Longest wait")
        st.caption(f"Ordered by priority, then waiting time. Nobody waits forever: P3→P2 after "
                   f"{RULES['STARVATION_SECONDS'][3] // 60} min, P2→P1 after "
                   f"{RULES['STARVATION_SECONDS'][2] // 60} min"
                   + (f" · **demo clock ×{clock}**" if clock != 1 else ""))
        if not q:
            st.success("No calls waiting.")
            return
        for c in q:
            with st.container(border=True):
                col = st.columns([0.9, 1.0, 4.4, 3.2, 0.9], vertical_alignment="center")
                col[0].markdown(priority_tile(c["eff"]), unsafe_allow_html=True)
                bumped = (f"<div class='cv-sub'>⏱️ was P{c['final_priority']}</div>"
                          if c["bumps"] else "")
                col[1].markdown(f"<div class='cv-sub'>Bed</div><div class='cv-bed'>{c['bed']}</div>"
                                f"<div class='cv-sub'>waiting {fmt_wait(c['waited'])}</div>{bumped}",
                                unsafe_allow_html=True)
                col[2].markdown(class_badge(c["request_class"], c.get("medication"), c.get("care_request"),
                                            c.get("distress_words"))
                                + f"<div class='cv-quote'>{quote(c)}</div>" + sentiment_chip(c),
                                unsafe_allow_html=True)
                col[3].markdown(f"<div class='cv-why'>{plain_reason(c, c['bumps'], c['waited'])}</div>",
                                unsafe_allow_html=True)
                if col[4].button("Done", key=f"ans_{c['qid']}", help="Mark answered",
                                 width="stretch"):
                    answer(c["qid"])
                    st.rerun(scope="fragment")

    queue_view()
    with st.expander("Audit log: every decision, append-only"):
        rows = audit_log().recent(100)
        if rows:
            df = pd.DataFrame(rows)[["id", "ts_utc", "mode", "bed", "request_class", "base_priority",
                                     "final_priority", "rules_fired", "transcript", "audio_sha256"]]
            st.dataframe(df, hide_index=True, width="stretch")
        else:
            st.caption("No decisions logged yet.")


# --------------------------------------------------------------------------- tab 2

def result_card(r: dict):
    with st.container(border=True):
        top = st.columns([1.1, 5], vertical_alignment="center")
        top[0].markdown(priority_tile(r["final_priority"], big=True), unsafe_allow_html=True)
        top[1].markdown(
            f"<div class='cv-sub'>Bed {r['bed']}{' · replayed (cached mode)' if r.get('cached') else ''}</div>"
            + class_badge(r["request_class"], r["medication"], r.get("care_request"), r.get("distress_words"))
            + f"<div class='cv-quote'>{quote(r)}</div>"
            f"<div class='cv-why'><b>Why:</b> {plain_reason(r)}</div>" + sentiment_chip(r),
            unsafe_allow_html=True)
        if r["request_class"] == EMERGENCY_BUTTON:
            st.caption(f"Sent to the nurse station as P1 — the alarm sounds there until a nurse "
                       f"marks bed {r['bed']} done.")
            return
        m = st.columns(3)
        m[0].metric("Response time", f"{r['latency_ms']['total'] / 1000:.1f} s")
        m[1].metric("Speech-to-text confidence", "trusted" if r["asr_trusted"] else "low",
                    help=f"Whisper log-probability {fmt(r['asr_confidence'], 2)} "
                         f"(trusted above {RULES['ASR_MIN']})")
        m[2].metric("Request confidence",
                    f"{r['class_confidence']:.0%}" if r["class_probs"] else "n/a",
                    help="ClinicalBERT's probability for its top class")

    with st.expander("How it decided"):
        left, right = st.columns(2)
        with left:
            if r["class_probs"]:
                st.markdown("**Request type** — ClinicalBERT probabilities")
                st.bar_chart(pd.Series(r["class_probs"], name="probability"), horizontal=True, height=180)
                if r.get("classifier_class") and r["classifier_class"] != r["request_class"]:
                    st.caption(f"The model said {r['classifier_class']}; a safety rule routed it to "
                               f"{r['request_class']} (the training data contains only symptoms).")
            else:
                st.markdown(f"**No words:** {r['speech_reason']}. Routed like a call-button press, "
                            "never dropped.")
            if r.get("sentiment"):
                st.markdown("**Sentiment of the words** — shown to the nurse, never changes priority")
                st.bar_chart(pd.Series(r["sentiment"], name="probability"), horizontal=True, height=200)
        with right:
            st.markdown("**Rules applied**")
            st.markdown(f"Base priority from the request: {priority_badge(r['base_priority'])} → "
                        f"final: {priority_badge(r['final_priority'])}", unsafe_allow_html=True)
            st.caption(r["reason"])
            st.markdown(f"**Speech check** (WavLM): "
                        f"{'speech present' if r['speech_present'] else 'no speech'}"
                        + (f" · non-speech score {r['non_speech_prob']:.2f}"
                           if r["non_speech_prob"] is not None else ""))
            st.markdown("**Time per stage** (ms)")
            lat = pd.Series(r["latency_ms"]).drop("total")
            st.bar_chart(lat[lat > 0], horizontal=True, height=220)


MIC_HELPER_HTML = """
<div style="font-family:sans-serif;font-size:14px;color:#0f172a">
  <button id="ask" style="background:#1d4ed8;color:#fff;border:0;border-radius:8px;padding:8px 14px;
    font-size:14px;font-weight:600;cursor:pointer;white-space:nowrap">🎙️ Allow microphone</button>
  <div id="state" style="margin-top:8px;font-weight:600"></div>
  <div id="help" style="margin-top:8px;color:#475569;font-size:13px;line-height:1.45"></div>
</div>
<script>
const st = document.getElementById('state'), help = document.getElementById('help');
const say = (icon, msg, tip) => { st.textContent = icon + ' ' + msg; help.innerHTML = tip || ''; };
const host = window.top === window ? location.hostname : (document.referrer ? new URL(document.referrer).hostname : location.hostname);
async function current() {
  if (!window.isSecureContext || !navigator.mediaDevices) {
    say('⛔', 'This address cannot use a microphone.',
        'Open the app at <b>http://localhost:8501</b> (not a 192.168… address).'); return; }
  try {
    const p = await navigator.permissions.query({name: 'microphone'});
    const show = () => p.state === 'granted' ? say('✅', 'Microphone allowed — use the recorder below.')
      : p.state === 'denied' ? say('⛔', 'Microphone blocked for this site.',
          'In Arc: click the icon left of the address bar → Microphone → <b>Allow</b>, then reload. ' +
          'If it is already allowed there: System Settings → Privacy &amp; Security → Microphone → turn on Arc, then quit and reopen Arc.')
      : say('ℹ️', 'Click the button: your browser will ask for permission.');
    show(); p.onchange = show;
  } catch (e) { say('ℹ️', 'Click the button: your browser will ask for permission.'); }
}
document.getElementById('ask').onclick = async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({audio: true});
    s.getTracks().forEach(t => t.stop());
    say('✅', 'Microphone allowed — use the recorder below.');
  } catch (e) {
    // Chromium (Arc, Chrome) says "Permission denied by system" when macOS blocks
    // the browser itself, and plain "Permission denied" when the site is blocked.
    if (e.name === 'NotAllowedError' && /system/i.test(e.message)) say('⛔', 'macOS is blocking Arc from the microphone.',
      'System Settings → Privacy &amp; Security → <b>Microphone</b> → switch <b>Arc</b> on (if Arc is not in the list, ' +
      'it has never asked: quit Arc with ⌘Q, reopen it, click this button again and accept the macOS pop-up). ' +
      'After switching it on, quit Arc fully with ⌘Q and reopen it — macOS only applies it on restart.' +
      '<br><span style="color:#94a3b8">(' + e.message + ')</span>');
    else if (e.name === 'NotAllowedError') say('⛔', 'Arc has blocked the microphone for this site.',
      'Click the <b>ⓘ</b> icon at the left of the address bar → <b>Microphone → Allow</b> (or Reset permission), then reload. ' +
      'If there is no Microphone entry there: Arc Settings → Privacy → Site settings → Microphone → remove localhost from "Not allowed".' +
      '<br><span style="color:#94a3b8">(' + e.message + ')</span>');
    else if (e.name === 'NotFoundError') say('⛔', 'No microphone found.', 'Plug in or enable a microphone, then try again.');
    else say('⛔', e.name + ': ' + e.message);
  }
};
current();
</script>
"""


def mic_permission_helper():
    """Asks the browser for the microphone on click, which makes it show its
    permission prompt, and says in plain words why recording would fail (blocked
    site, macOS permission off, no mic, non-localhost address). Runs in a component
    iframe; Streamlit's iframes are allowed to use the microphone."""
    import streamlit.components.v1 as components
    components.html(MIC_HELPER_HTML, height=200, scrolling=True)


def demo_label(fname: str) -> str:
    stem = Path(fname).stem
    num, _, rest = stem.partition("_")
    what, _, who = rest.partition("__")
    return f"{num} · {what.replace('_', ' ').capitalize()}" + (f" — {who.capitalize()}" if who else "")


def tab_submit(cached_mode: bool):
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.markdown("#### Patient")
        eb_bed = st.selectbox("Bed", CONFIG["beds"], index=CONFIG["beds"].index("14"), key="eb_bed")
        # A click callback, not `if st.button(...)`: the queue refreshes every second and
        # a returned True was replayed on those reruns (one press made a call per second).
        st.button("🚨  EMERGENCY — I need help now", key="emergency_btn", width="stretch",
                  on_click=on_emergency_click)
        st.caption("One press, no recording: an immediate P1 call that sounds at the nurse station.")
        st.divider()
        st.markdown("#### Or speak a request")
        source = st.radio("Audio", ["Demo clip", "Record", "Upload a file"], horizontal=True,
                          label_visibility="collapsed")
        name = data = None
        default_bed = "14"
        if source == "Record":
            mic_permission_helper()
            rec = st.audio_input("Press to record the patient's request",
                                 sample_rate=CONFIG["mic_sample_rate"])
            if rec:
                name, data = f"mic_{time.strftime('%H%M%S')}.wav", rec.getvalue()
            if cached_mode:
                st.warning("Cached mode replays prepared clips only; switch it off to record.")
        elif source == "Upload a file":
            up = st.file_uploader("Audio file", type=CONFIG["audio_types"])
            if up:
                name, data = up.name, up.getvalue()
        else:
            demo = sorted(p.name for p in CONFIG["demo_dir"].glob("*")
                          if p.suffix.lstrip(".").lower() in CONFIG["audio_types"])
            if not demo:
                st.warning(f"No clips in {CONFIG['demo_dir'].relative_to(ROOT)}.")
            else:
                pick = st.selectbox("Demo clip", demo, key="demo_clip", format_func=demo_label)
                name, data = pick, (CONFIG["demo_dir"] / pick).read_bytes()
                who = Path(pick).stem.partition("__")[2]
                default_bed = SPEAKER_BED.get(who, default_bed)
                st.audio(data)
        bed = st.selectbox("Bed", CONFIG["beds"], index=CONFIG["beds"].index(default_bed),
                           key=f"bed_{default_bed}")
        go = st.button("Send call to nurse station", type="primary", disabled=data is None,
                       width="stretch")
    with right:
        if go:
            with st.spinner("Listening and deciding..."):
                try:
                    r = run_call(name, data, bed, cached_mode)
                except Exception as e:  # keep the demo standing; point at cached mode
                    st.error(f"Processing failed: {e}. Switch on cached mode in the sidebar.")
                    r = None
            if r:
                st.session_state["last_result"] = r
        if "last_result" in st.session_state:
            st.markdown("#### Result")
            result_card(st.session_state["last_result"])
        else:
            st.markdown("#### Result")
            st.info("Pick a demo clip (or record one) and send it. The decision appears here and "
                    "the call joins the Nurse Station queue.")


# --------------------------------------------------------------------------- tab 3

def _json(name):
    try:
        return json.loads((CONFIG["results_dir"] / name).read_text())
    except FileNotFoundError:
        return None


def _csv(name, **kw):
    p = CONFIG["results_dir"] / name
    return pd.read_csv(p, **kw) if p.exists() else None


def tab_results():
    st.markdown("#### How it works")
    steps = [("Capture", "Patient speaks at the bedside"), ("Whisper", "Speech → text"),
             ("ClinicalBERT", "What they need"), ("Safety rules", "How urgent, and why"),
             ("Nurse station", "Prioritised queue")]
    st.markdown("<span class='cv-arrow'>→</span>".join(
        f"<span class='cv-step'><b>{a}</b>{b}</span>" for a, b in steps), unsafe_allow_html=True)
    st.caption("A speech check (WavLM) tells real requests from silence and noise. Priority comes "
               "from written rules, not a black-box model: every decision has a reason and is logged.")

    st.markdown("#### Measured accuracy")
    st.caption("Emergency recall is the number that matters: a missed emergency and a misrouted "
               "water request are not the same mistake.")
    e2e = _json("end_to_end_v3.json")
    tb = _json("text_branch_final.json")
    c = st.columns(4)
    if e2e:
        kpi(c[0], f"{e2e['HEADLINE_emergency_under_triage']:.1%}", "Emergencies under-triaged")
        kpi(c[1], f"{e2e['emergency_within_p2']:.1%}", "Emergencies reaching P1/P2")
        kpi(c[2], f"{e2e['HEADLINE_under_triage']:.1%}", "All calls under-triaged")
        kpi(c[3], f"{e2e['latency_median_ms'] / 1000:.1f} s", "Median response time")
        st.caption(f"Full system, audio in → priority out, on {e2e['n_test_clips']:,} recordings of "
                   f"{e2e['n_test_phrases']} sentences it never saw in training.")
    elif tb:
        kpi(c[0], f"{tb['clinicalbert_clean']['emergency_recall']:.1%}", "Emergency recall (text)")
        kpi(c[1], f"{tb['clinicalbert_on_asr']['emergency_recall']:.1%}", "Emergency recall (from speech)")
        kpi(c[2], f"{tb['wer_full_corpus']:.1%}", "Speech-to-text word error")
        kpi(c[3], "706", "Distinct patient sentences")
        st.caption("Medical Speech corpus, 6,661 recordings, 5-fold cross-validation grouped by "
                   "sentence. Full end-to-end numbers appear here when the evaluation finishes.")

    own = _csv("own_voice_check.csv")
    if own is not None:
        st.markdown("#### Tested on our own voices")
        calm = own[own.kind == "calm request"]
        emer = own[own.kind == "emergency"]
        c = st.columns(4)
        kpi(c[0], f"{(emer.final_priority == 1).sum()}/{len(emer)}", "Emergency calls → P1")
        kpi(c[1], f"{(calm.final_priority == 1).sum()}/{len(calm)}", "Everyday requests → P1 (false alarms)")
        kpi(c[2], own.speaker.nunique(), "Speakers")
        kpi(c[3], f"{own.latency_s.median():.1f} s", "Median response time")
        show = own[["speaker", "transcript", "request_class", "final_priority"]].rename(
            columns={"request_class": "routed to", "final_priority": "priority"})
        st.dataframe(show, hide_index=True, width="stretch", height=280)

    fig = CONFIG["results_dir"] / "noise_curve_heldout.png"
    if fig.exists():
        st.markdown("#### In a noisy ward")
        st.image(str(fig), width="stretch")

    with st.expander("Known limitations"):
        st.markdown(
            "- **Distress in the tone of voice is not detected.** Four approaches were tested on our "
            "own recordings and none caught all three speakers without false alarms, so it is off. "
            "Distress is detected from the words (\"please come fast\", \"oh no!\") and emergencies "
            "from what is said.\n"
            "- **The training data contains only symptoms.** Everyday requests (window, light, "
            "toilet, medication) are handled by explicit, tested rules rather than the model.\n"
            "- **Speech-to-text can invent words from loud machine noise.** The speech check "
            "catches most of it; a loud fan near the microphone is the case to avoid.\n"
            "- **Sentiment is informational.** It is shown to the nurse and never changes priority.")


# --------------------------------------------------------------------------- layout

st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Presenter controls")
    cached_mode = st.toggle("Cached mode", value=False, key="cached_mode",
                            help="Replay precomputed results for the demo clips instead of running "
                                 "the models. The backup if anything fails on stage.")
    n_cached = len(load_cache())
    st.caption(f"{n_cached} demo clip(s) ready for cached mode" if n_cached else "No cached results yet.")
    st.select_slider("Demo clock", CONFIG["demo_clock"], value=1, key="demo_clock",
                     format_func=lambda v: f"×{v}",
                     help="Speeds up waiting time so the 'nobody waits forever' timer is visible.")
    st.toggle("Sentiment", value=True, key="emotion_on", on_change=apply_emotion_toggle,
              help="Text sentiment, shown for every call; never affects priority. Switch off if "
                   "the machine is short of memory.")
    st.toggle("Alert sound", value=True, key="alert_sound",
              help="Beep at the nurse station when a new P1 call arrives.")
    st.toggle("Keep models warm", value=False, key="keep_warm",
              help=f"Runs a dummy inference every {CONFIG['keep_warm_s']} s so the weights are "
                   "not swapped out between calls. Only helps once the machine has free RAM.")
    if st.button("Reset demo", width="stretch",
                 help="Clears waiting calls and per-bed voice baselines. The audit log is kept."):
        queue_db().execute("DELETE FROM queue")
        queue_db().execute("CREATE TABLE IF NOT EXISTS prosody_baseline (id INTEGER PRIMARY KEY)")
        queue_db().execute("DELETE FROM prosody_baseline")
        queue_db().commit()
        st.session_state.pop("last_result", None)
        st.rerun()

    # Keep-warm: touching every loaded model's weights stops the OS paging them out
    # between demo calls (measured: 1.5 s warm vs ~10 s after 60 s idle on the 8 GB
    # machine). It cannot create memory: with RAM full it just evicts something else,
    # so it is off by default. Runs only while a browser session is open.
    @st.fragment(run_every=CONFIG["keep_warm_s"])
    def keep_warm_ping():
        if st.session_state.get("keep_warm") and not st.session_state.get("cached_mode"):
            ms = load_pipeline().keep_warm()
            st.caption(f"Keep-warm ping {time.strftime('%H:%M:%S')}: {ms:.0f} ms")

    keep_warm_ping()

header(cached_mode)
t1, t2, t3 = st.tabs(["🏥 Nurse Station", "🎙️ Submit a Call", "📊 Results"])
with t1:
    tab_nurse_station()
with t2:
    tab_submit(cached_mode)
with t3:
    tab_results()
