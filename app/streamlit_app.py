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
from src.mapping import CLASS_BASE_PRIORITY
from src.rules import CONFIG as RULES
from src.rules import UNCLASSIFIED, starvation_priority

CONFIG = {
    "model_dir": ROOT / "models",
    "demo_dir": ROOT / "app" / "demo_clips",
    "cache_file": ROOT / "app" / "cached_results.json",
    "queue_db": ROOT / "app" / "calls.db",
    "audit_db": ROOT / "app" / "audit_log.db",
    "results_dir": ROOT / "results",
    "queue_refresh_s": 1,          # the waiting timer ticks every second
    "mic_sample_rate": 16000,      # capture at the pipeline's rate; process() still resamples
    # Everything phones, QuickTime and browsers produce. All decode through
    # src/preprocess.py (libsndfile, else ffmpeg); a video's audio track is used.
    "audio_types": ["wav", "mp3", "m4a", "mp4", "mov", "qta", "aac", "flac", "ogg", "opus",
                    "webm", "aiff", "aif", "caf", "3gp", "amr"],
    "beds": [str(b) for b in range(1, 31)],
    "demo_clock": [1, 10, 60],     # waiting time multiplier, so starvation is visible on stage
    "keep_warm_s": 30,             # dummy-inference interval when the keep-warm switch is on
}

PRIORITY_STYLE = {
    1: ("#b71c1c", "#ffffff", "P1 immediate"),
    2: ("#e65100", "#ffffff", "P2 urgent"),
    3: ("#f9a825", "#1a1a1a", "P3 standard"),
    4: ("#2e7d32", "#ffffff", "P4 routine"),
}
UNCLASSIFIED_STYLE = ("#757575", "#ffffff")

# Plain-language versions of the rule ids in rules.Decision.rules_fired.
RULE_WORDS = {
    "R0_no_speech": "No words or voice detected: treated like a plain call-button press",
    "R0_audio_error": "Recording missing or unreadable: treated like a plain call-button press",
    "R1_distress_untrusted_transcript": "Sounds distressed and the words are unclear: go now",
    "R2_distress": "Voice sounds distressed: raised to immediate",
    "R4_emergency_floor": "Emergency request: kept urgent however calm it sounds",
    "R5_care_request": "Asked for help to the toilet or to move: raised to urgent",
}

st.set_page_config(page_title="Care-Voice AI", page_icon="🔔", layout="wide")


# --------------------------------------------------------------------------- display helpers

def _badge(text: str, bg: str, fg: str, small: bool = False) -> str:
    size = "0.75em" if small else "0.85em"
    return (f"<span style='background:{bg};color:{fg};padding:2px 9px;border-radius:10px;"
            f"font-weight:600;font-size:{size};white-space:nowrap'>{text}</span>")


def priority_badge(p: int, short: bool = False) -> str:
    bg, fg, label = PRIORITY_STYLE[p]
    return _badge(f"P{p}" if short else label, bg, fg)


def class_badge(request_class: str, medication: list | None = None,
                care: list | None = None) -> str:
    # Unclassified is a pipeline sentinel, not one of the four trained classes: grey,
    # never a priority colour.
    if request_class == UNCLASSIFIED:
        html = _badge(UNCLASSIFIED, *UNCLASSIFIED_STYLE)
    else:
        bg, fg, _ = PRIORITY_STYLE[CLASS_BASE_PRIORITY[request_class]]
        html = _badge(request_class, bg, fg)
    if medication:
        # Keyword rule, not a class (src/mapping.py): a sub-label under the class.
        html += "<br>" + _badge("💊 medication", "#4527a0", "#ffffff", small=True)
    if care:
        # Keyword rule (src/mapping.py), floors the call at P2 (rules.py Rule 5).
        html += "<br>" + _badge("🚻 care request", "#00695c", "#ffffff", small=True)
    return html


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
        lines.append(f"⏱️ Waiting {int(waited // 60)} min: bumped up from P{r['final_priority']}")
    return "<br>".join(lines)


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
    pipe = CareVoicePipeline(model_dir=str(CONFIG["model_dir"]), device="cpu", load_emotion=False)
    st.session_state["pipeline_loaded"] = True
    return pipe


def apply_emotion_toggle():
    """Load or free the display-only emotion model in the running pipeline. Does
    nothing before the pipeline exists: it is applied at the first call instead."""
    if st.session_state.get("pipeline_loaded"):
        load_pipeline().set_emotion(st.session_state.get("emotion_on", False))


def load_cache() -> dict:
    try:
        return json.loads(CONFIG["cache_file"].read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


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
        pipe.set_emotion(st.session_state.get("emotion_on", False))
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


# --------------------------------------------------------------------------- tab 1

def tab_nurse_station():
    widths = [0.9, 0.5, 0.7, 1.3, 3.0, 3.2, 0.5]

    @st.fragment(run_every=CONFIG["queue_refresh_s"])
    def queue_view():
        clock = st.session_state.get("demo_clock", 1)
        q = waiting_queue(clock)
        st.caption(f"{len(q)} waiting · sorted by priority, then waiting time · starvation timer "
                   f"P4→P3 after {RULES['STARVATION_SECONDS'][4] // 60} min, P3→P2 after "
                   f"{RULES['STARVATION_SECONDS'][3] // 60} min, P2→P1 after "
                   f"{RULES['STARVATION_SECONDS'][2] // 60} min"
                   + (f" · **demo clock ×{clock}**" if clock != 1 else ""))
        if not q:
            st.success("No calls waiting.")
            return
        for h, t in zip(st.columns(widths), ["Priority", "Bed", "Waiting", "Request", "Transcript",
                                              "Why", ""]):
            h.markdown(f"**{t}**")
        for c in q:
            row = st.columns(widths, vertical_alignment="center")
            badge = priority_badge(c["eff"], short=True)
            if c["bumps"]:
                badge += f" <span style='font-size:0.75em'>⏱️ was P{c['final_priority']}</span>"
            row[0].markdown(badge, unsafe_allow_html=True)
            row[1].markdown(f"**{c['bed']}**")
            row[2].markdown(f"`{fmt_wait(c['waited'])}`")
            row[3].markdown(class_badge(c["request_class"], c.get("medication"), c.get("care_request")),
                            unsafe_allow_html=True)
            row[4].markdown(c["transcript"] or "*(no words)*")
            row[5].markdown(plain_reason(c, c["bumps"], c["waited"]), unsafe_allow_html=True)
            if row[6].button("✓", key=f"ans_{c['qid']}", help="Mark answered"):
                answer(c["qid"])
                st.rerun(scope="fragment")

    queue_view()
    if st.button("Clear queue", help="Clears waiting calls only. The audit log cannot be cleared."):
        queue_db().execute("DELETE FROM queue")
        queue_db().commit()
        st.rerun()
    with st.expander("Audit log (append-only, read-only view)"):
        rows = audit_log().recent(100)
        if rows:
            df = pd.DataFrame(rows)[["id", "ts_utc", "mode", "bed", "request_class", "base_priority",
                                     "final_priority", "rules_fired", "distress", "non_speech_prob",
                                     "text_anger", "transcript", "audio_sha256", "thresholds"]]
            st.dataframe(df, hide_index=True, width="stretch")
        else:
            st.caption("No decisions logged yet.")


# --------------------------------------------------------------------------- tab 2

def render_breakdown(r: dict):
    c = st.columns(5)
    c[0].markdown(f"**Final**<br>{priority_badge(r['final_priority'])}", unsafe_allow_html=True)
    c[1].markdown(f"**Base**<br>{priority_badge(r['base_priority'])}", unsafe_allow_html=True)
    c[2].markdown(f"**Request**<br>{class_badge(r['request_class'], r['medication'], r.get('care_request'))}",
                  unsafe_allow_html=True)
    c[3].markdown(f"**Rule fired**<br>{', '.join(r['rules_fired']) or 'none'}", unsafe_allow_html=True)
    c[4].markdown(f"**Latency**<br>{r['latency_ms']['total']:.0f} ms", unsafe_allow_html=True)
    st.markdown(f"**In plain words:** {plain_reason(r).replace('<br>', '; ')}")
    st.caption(f"Rule trace: {r['reason']}")
    if r.get("cached"):
        st.caption("Replayed from cached_results.json (cached mode).")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Text branch")
        st.markdown(f"**Transcript:** {r['transcript'] or '*(empty)*'}")
        if not r["speech_present"]:
            st.info(f"No speech: {r['speech_reason']}. Text branch skipped.")
            if r.get("asr_heard"):
                st.caption(f"Whisper heard {r['asr_heard']!r}; discarded as an untrusted "
                           "hallucination.")
        st.markdown(f"**ASR confidence:** {fmt(r['asr_confidence'])} (ASR_MIN {RULES['ASR_MIN']}) — "
                    f"{'trusted' if r['asr_trusted'] else '**not trusted**'}")
        if r["class_probs"]:
            st.markdown("**Request type** (ClinicalBERT)")
            st.bar_chart(pd.Series(r["class_probs"], name="probability"), horizontal=True, height=170)
        else:
            st.markdown(f"**Request type:** {class_badge(UNCLASSIFIED)} (no transcript, ClinicalBERT "
                        "not run)", unsafe_allow_html=True)
        st.markdown(f"**Care request:** {', '.join(r.get('care_request') or []) or 'none'} "
                    "<span style='font-size:0.8em'>(keyword rule; floors at P2)</span>",
                    unsafe_allow_html=True)
        st.markdown(f"**Medication tag:** {', '.join(r['medication']) or 'none'} "
                    "<span style='font-size:0.8em'>(keyword rule, not a class)</span>",
                    unsafe_allow_html=True)
        if r["sentiment"]:
            st.markdown(f"**Sentiment:** {r['sentiment_label']} · anger {fmt(r['text_anger'], 2)} "
                        "— *display only, never affects priority*")
            st.bar_chart(pd.Series(r["sentiment"], name="probability"), horizontal=True, height=200)
        else:
            st.markdown("**Sentiment:** " + ("n/a (no transcript)" if r.get("sentiment_enabled")
                                             else "off (sidebar: Text emotion)"))
    with right:
        st.markdown("##### Acoustic branch")
        if r["distress"] is None:
            st.caption("Not run: the energy pre-filter found no signal (digital silence).")
        else:
            st.markdown(f"**Distress gate:** {fmt(r['distress'], 4)} (DISTRESS_HIGH "
                        f"{RULES['DISTRESS_HIGH']}) {'— 🔴 **fired**' if r['distress_fired'] else ''}")
            st.progress(min(1.0, r["distress"]))
            st.markdown(f"**Non-speech:** {fmt(r['non_speech_prob'], 4)} (NON_SPEECH_HIGH "
                        f"{RULES['NON_SPEECH_HIGH']}) — speech guard: "
                        f"{'speech present' if r['speech_present'] else '**no speech**'}")
            st.progress(min(1.0, r["non_speech_prob"]))
            st.markdown(f"**Fear:** {fmt(r['arousal'], 4)} — *display only, never affects priority*")
            st.progress(min(1.0, r["arousal"]))
        st.markdown("##### Latency per stage (ms)")
        lat = pd.Series(r["latency_ms"]).drop("total")
        st.bar_chart(lat[lat > 0], horizontal=True, height=230)


def tab_submit(cached_mode: bool):
    bed = st.selectbox("Bed", CONFIG["beds"], index=13)
    source = st.radio("Audio", ["Record", "Upload a file", "Demo clip"], horizontal=True)
    name = data = None
    if source == "Record":
        rec = st.audio_input("Press to record the patient's request",
                             sample_rate=CONFIG["mic_sample_rate"])
        if rec:
            name, data = f"mic_bed{bed}_{time.strftime('%H%M%S')}.wav", rec.getvalue()
        if cached_mode:
            st.warning("Cached mode cannot process a new recording; switch it off.")
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
            pick = st.selectbox("Clip", demo, key="demo_clip")
            name, data = pick, (CONFIG["demo_dir"] / pick).read_bytes()
            st.audio(data)

    if st.button("Submit call", type="primary", disabled=data is None):
        with st.spinner("Processing..."):
            try:
                r = run_call(name, data, bed, cached_mode)
            except Exception as e:  # keep the demo standing; point at cached mode
                st.error(f"Pipeline failed: {e}. Switch on cached mode in the sidebar.")
                r = None
        if r:
            st.session_state["last_result"] = r
    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        st.divider()
        st.markdown(f"#### Bed {r['bed']} — {r['audio_path']} · audit #{r['audit_id']}")
        render_breakdown(r)


# --------------------------------------------------------------------------- tab 3

def _json(name):
    try:
        return json.loads((CONFIG["results_dir"] / name).read_text())
    except FileNotFoundError:
        return None


def _csv(name, **kw):
    p = CONFIG["results_dir"] / name
    return pd.read_csv(p, **kw) if p.exists() else None


def _fig(name, caption):
    p = CONFIG["results_dir"] / "figures" / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


def tab_results():
    st.markdown("Emergency recall is the headline metric, not accuracy: a missed emergency and a "
                "misrouted water request are not the same error.")

    tb = _json("text_branch_final.json")
    if tb:
        st.markdown("#### Text branch — Emergency recall")
        st.caption("Per recording (n = 6,661), 5-fold CV grouped by phrase.")
        c = st.columns(4)
        c[0].metric("ClinicalBERT, clean text", f"{tb['clinicalbert_clean']['emergency_recall']:.3f}")
        c[1].metric("ClinicalBERT, on Whisper output", f"{tb['clinicalbert_on_asr']['emergency_recall']:.3f}")
        c[2].metric("TF-IDF baseline", f"{tb['tfidf_phrase_level']['emergency_recall']:.3f}")
        c[3].metric("Whisper small WER", f"{tb['wer_full_corpus']:.1%}")
        st.caption(f"Missed emergencies: {tb['clinicalbert_clean']['emergency_fn']} on clean text, "
                   f"{tb['clinicalbert_on_asr']['emergency_fn']} on Whisper output.")
    rep = _csv("clinicalbert_report.csv", index_col=0)
    if rep is not None:
        st.markdown("**ClinicalBERT per-class** (recall is the column that matters; per unique phrase, n = 706)")
        st.dataframe(rep.round(3), width="stretch")

    v2 = _json("head_refit_v2.json")
    if v2:
        st.markdown("#### Acoustic branch (v2: unpadded, deduplicated, noise and silence negatives)")
        c = st.columns(4)
        c[0].metric("Distress AUC, held out", f"{v2['distress_heldout_auc_all_negatives']:.3f}",
                    help=f"Within VIVAE only (pain vs other vocalisations): {v2['distress_withincorpus_auc']:.3f}")
        c[1].metric("Distress recall", f"{v2['distress_recall']:.1%}", help=f"at DISTRESS_HIGH = {v2['DISTRESS_HIGH_new']}")
        c[2].metric("False alarms, medical speech", f"{v2['distress_fa_medical']:.2%}",
                    help=f"ward noise {v2['distress_fa_ward_noise']:.1%}, silence {v2['distress_fa_silence']:.0%}")
        c[3].metric("Fear recall at 2% false alarms", f"{v2['arousal_recall_at_2pct_medical_fa']:.1%}",
                    help="Too low to escalate on: fear is displayed, never used")
        curve = _csv("operating_curve_distress_v2.csv")
        if curve is not None:
            left, right = st.columns(2)
            with left:
                st.markdown("**Distress operating curve** (held out; recall against false-alarm rate)")
                st.line_chart(curve.set_index("recall")[["fa_medical", "fa_ward_noise", "fa_vivae_other"]],
                              height=260)
            with right:
                st.markdown(f"**At DISTRESS_HIGH = {RULES['DISTRESS_HIGH']}**")
                row = curve[curve.threshold.round(4) == round(RULES["DISTRESS_HIGH"], 4)]
                st.dataframe(row.T.set_axis(["value"], axis=1), width="stretch")
        groan = _csv("groan_check_v2.csv")
        if groan is not None:
            st.markdown(f"**Groan check:** {(groan.P == 1).sum()}/{len(groan)} VIVAE pain/fear clips "
                        f"reached P1 through the full pipeline; {(groan.ns >= RULES['NON_SPEECH_HIGH']).sum()} "
                        "of them scored as non-speech and were let through by the distress key of the "
                        "speech guard. In-sample for the gate (held-out recall is "
                        f"{v2['distress_recall']:.1%}).")
        _fig("padding_diagnostic.png", "Padding diagnostic: why the v1 numbers were superseded")

    with st.expander("Superseded: v1 acoustic results (padded features)"):
        st.error("These numbers measured padding and corpus identity, not distress: clip duration "
                 "alone separated the training corpora at AUC 0.9965 (results/padding_diagnostic.json). "
                 "Kept for the record; do not quote.")
        g1 = _json("distress_gate_results.json")
        e2e = _json("end_to_end.json")
        if g1:
            c = st.columns(3)
            c[0].metric("v1 distress AUC vs calm speech", g1["auc_vs_calm_speech"])
            c[1].metric("v1 recall", f"{g1['operating_point']['recall']:.0%}")
            c[2].metric("v1 false alarms", f"{g1['operating_point']['false_alarm']:.1%}")
        if e2e:
            st.caption(f"v1 end to end: Emergency reaching P1/P2 {e2e['emergency_reached_P1_P2']:.3f}, "
                       f"escalated by acoustics {e2e['escalated_by_acoustics']} (v1 heads, arousal rule).")
        cols = st.columns(3)
        for col, (f, cap) in zip(cols, [("distress_curve.png", "v1 distress curve"),
                                         ("operating_curve.png", "v1 arousal curve"),
                                         ("gap_check.png", "v1 domain-gap check")]):
            with col:
                _fig(f, cap)

    if not any(f.name.startswith("noise") for f in (CONFIG["results_dir"] / "figures").glob("*.png")):
        st.info("Noise-degradation figure (notebook 07) not produced yet.")


# --------------------------------------------------------------------------- layout

with st.sidebar:
    st.markdown("## 🔔 Care-Voice AI")
    cached_mode = st.toggle("Cached mode", value=False, key="cached_mode",
                            help="Replay precomputed results instead of running the models live.")
    n_cached = len(load_cache())
    st.caption(f"{n_cached} clip(s) in cached_results.json" if n_cached else "No cached results yet.")
    st.select_slider("Demo clock", CONFIG["demo_clock"], value=1, key="demo_clock",
                     format_func=lambda v: f"×{v}",
                     help="Speeds up waiting time so the starvation timer is visible on stage.")
    st.toggle("Text emotion (display only)", value=False, key="emotion_on",
              on_change=apply_emotion_toggle,
              help="Loads the 313 MB emotion model. It never affects priority, so it is off "
                   "by default to keep the demo machine out of swap.")
    st.toggle("Keep models warm", value=False, key="keep_warm",
              help=f"Runs a dummy inference every {CONFIG['keep_warm_s']} s so the weights are "
                   "not swapped out between calls. Only helps once the machine has free RAM; "
                   "under memory pressure it pushes the browser out instead.")

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
    st.caption("Escalation is an auditable rule (src/rules.py), not a trained fusion model.")

t1, t2, t3 = st.tabs(["Nurse Station", "Submit a Call", "Results"])
with t1:
    tab_nurse_station()
with t2:
    tab_submit(cached_mode)
with t3:
    tab_results()
