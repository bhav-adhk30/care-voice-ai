"""
Care-Voice AI demo: nurse station, call submission, and results.

    HF_HUB_OFFLINE=1 streamlit run app/streamlit_app.py

Imports CareVoicePipeline from src/pipeline.py, the same class the end-to-end
evaluation uses (CLAUDE.md section 2.7). There is no second implementation of the
pipeline here: this file only stores, orders and displays what process() returns.
Retrained heads (notebook 10) replace two .joblib files and need no change here.

Cached mode replays app/cached_results.json (built by scripts/build_demo_cache.py)
so the demo still runs if a model fails to load on the day.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.mapping import CLASS_BASE_PRIORITY
from src.rules import CONFIG as RULES
from src.rules import UNCLASSIFIED, apply_starvation_timer

CONFIG = {
    "model_dir": ROOT / "models",
    "demo_dir": ROOT / "app" / "demo_clips",
    "cache_file": ROOT / "app" / "cached_results.json",
    "db_file": ROOT / "app" / "calls.db",
    "results_dir": ROOT / "results",
    "refresh_seconds": 5,
    "audio_types": ["wav", "mp3", "flac", "ogg", "m4a", "aiff", "aif"],
    "beds": [str(b) for b in range(1, 31)],
}

PRIORITY_STYLE = {
    1: ("#b71c1c", "#ffffff", "P1 immediate"),
    2: ("#e65100", "#ffffff", "P2 urgent"),
    3: ("#f9a825", "#1a1a1a", "P3 standard"),
    4: ("#2e7d32", "#ffffff", "P4 routine"),
}
UNCLASSIFIED_STYLE = ("#757575", "#ffffff")

st.set_page_config(page_title="Care-Voice AI", page_icon="🔔", layout="wide")


# --------------------------------------------------------------------------- badges

def _badge(text: str, bg: str, fg: str) -> str:
    return (f"<span style='background:{bg};color:{fg};padding:2px 10px;border-radius:10px;"
            f"font-weight:600;font-size:0.85em;white-space:nowrap'>{text}</span>")


def priority_badge(p: int, short: bool = False) -> str:
    bg, fg, label = PRIORITY_STYLE[p]
    return _badge(f"P{p}" if short else label, bg, fg)


def class_badge(request_class: str) -> str:
    # Unclassified is a pipeline sentinel, not one of the four trained classes: grey,
    # never a priority colour.
    if request_class == UNCLASSIFIED:
        return _badge(UNCLASSIFIED, *UNCLASSIFIED_STYLE)
    bg, fg, _ = PRIORITY_STYLE[CLASS_BASE_PRIORITY[request_class]]
    return _badge(request_class, bg, fg)


def fmt_wait(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# --------------------------------------------------------------------------- storage

@st.cache_resource
def db() -> sqlite3.Connection:
    con = sqlite3.connect(CONFIG["db_file"], check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS calls (
        id INTEGER PRIMARY KEY, bed TEXT, submitted_at REAL, source TEXT,
        result TEXT, status TEXT DEFAULT 'waiting', answered_at REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY, ts REAL, call_id INTEGER, event TEXT, detail TEXT)""")
    con.commit()
    return con


def audit(call_id, event: str, detail: str = ""):
    db().execute("INSERT INTO audit (ts, call_id, event, detail) VALUES (?,?,?,?)",
                 (time.time(), call_id, event, detail))
    db().commit()


def add_call(bed: str, source: str, result: dict) -> int:
    cur = db().execute("INSERT INTO calls (bed, submitted_at, source, result) VALUES (?,?,?,?)",
                       (bed, time.time(), source, json.dumps(result)))
    db().commit()
    audit(cur.lastrowid, "submitted",
          f"bed {bed}, P{result['final_priority']}, {result['request_class']}, {result['reason']}")
    return cur.lastrowid


def answer_call(call_id: int):
    db().execute("UPDATE calls SET status='answered', answered_at=? WHERE id=?", (time.time(), call_id))
    db().commit()
    audit(call_id, "answered")


def effective_priority(final_priority: int, waited: float) -> tuple[int, list[int]]:
    """Walk the starvation timer from rules.py: each level is held for its
    STARVATION_SECONDS before the call is bumped, so a P4 is not starved forever."""
    p, t, bumps = final_priority, waited, []
    while True:
        new_p, bumped = apply_starvation_timer(p, t)
        if not bumped:
            return p, bumps
        t -= RULES["STARVATION_SECONDS"][p]
        bumps.append(p)
        p = new_p


def waiting_queue() -> list[dict]:
    now, heap = time.time(), []
    rows = db().execute("SELECT id, bed, submitted_at, source, result FROM calls "
                        "WHERE status='waiting'").fetchall()
    for cid, bed, ts, source, res in rows:
        r = json.loads(res)
        eff, bumps = effective_priority(r["final_priority"], now - ts)
        heapq.heappush(heap, (eff, ts, cid, {"id": cid, "bed": bed, "waited": now - ts,
                                             "source": source, "eff": eff, "bumps": bumps, **r}))
    return [heapq.heappop(heap)[3] for _ in range(len(heap))]


# --------------------------------------------------------------------------- pipeline

@st.cache_resource(show_spinner="Loading models (Whisper, ClinicalBERT, WavLM)...")
def load_pipeline():
    from src.pipeline import CareVoicePipeline
    return CareVoicePipeline(model_dir=str(CONFIG["model_dir"]), device="cpu")


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
        return {**r, "bed": bed, "cached": True}
    pipe = load_pipeline()
    suffix = Path(name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        r = pipe.process(tmp, bed=bed)
    finally:
        os.unlink(tmp)
    r["audio_path"] = name
    return {**r, "cached": False}


# --------------------------------------------------------------------------- views

def render_breakdown(r: dict):
    c = st.columns([1, 1, 1, 1])
    c[0].markdown(f"**Final**<br>{priority_badge(r['final_priority'])}", unsafe_allow_html=True)
    c[1].markdown(f"**Base**<br>{priority_badge(r['base_priority'])}", unsafe_allow_html=True)
    c[2].markdown(f"**Request class**<br>{class_badge(r['request_class'])}", unsafe_allow_html=True)
    c[3].markdown(f"**Escalated**<br>{'⚠️ yes' if r['escalated'] else 'no'}", unsafe_allow_html=True)

    st.markdown(f"**Reason:** {r['reason']}")
    if r.get("cached"):
        st.caption("Replayed from cached_results.json (cached mode).")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Text branch")
        if not r["speech_present"]:
            st.info(f"No speech detected ({r['speech_reason']}). Both branches skipped.")
        st.markdown(f"**Transcript:** {r['transcript'] or '*(empty)*'}")
        conf = r["asr_confidence"]
        st.markdown(f"**ASR confidence:** {'n/a' if conf is None else f'{conf:.3f}'} "
                    f"(threshold {RULES['ASR_MIN']}) — "
                    f"{'trusted' if r['asr_trusted'] else '**not trusted**'}")
        if r["class_probs"]:
            probs = pd.Series(r["class_probs"], name="probability").sort_values(ascending=False)
            st.bar_chart(probs, horizontal=True, height=180)
        else:
            st.markdown(f"Class probabilities: none — {class_badge(UNCLASSIFIED)} "
                        "(empty transcript, ClinicalBERT not run)", unsafe_allow_html=True)
    with right:
        st.markdown("##### Acoustic branch")
        for label, key, thr, fired in (("Arousal (P(fear))", "arousal", RULES["AROUSAL_HIGH"], r["arousal_fired"]),
                                       ("Distress gate", "distress", RULES["DISTRESS_HIGH"], r["distress_fired"])):
            st.markdown(f"**{label}:** {r[key]:.4f} (threshold {thr}) "
                        f"{'— 🔴 **fired**' if fired else ''}")
            st.progress(min(1.0, r[key]))
        lat = pd.Series(r["latency_ms"], name="ms")
        st.markdown(f"##### Latency: {lat['total']:.0f} ms")
        st.dataframe(lat.drop("total").to_frame(), height=320)


def tab_nurse_station():
    @st.fragment(run_every=CONFIG["refresh_seconds"])
    def queue_view():
        q = waiting_queue()
        st.caption(f"{len(q)} waiting · refreshes every {CONFIG['refresh_seconds']} s · "
                   f"starvation timer {RULES['STARVATION_SECONDS']} s per level")
        if not q:
            st.success("No calls waiting.")
            return
        widths = [0.6, 0.5, 0.7, 1.3, 3.4, 2.8, 0.6]
        hdr = st.columns(widths)
        for h, t in zip(hdr, ["Priority", "Bed", "Waiting", "Request", "Transcript", "Escalation", ""]):
            h.markdown(f"**{t}**")
        for c in q:
            row = st.columns(widths)
            row[0].markdown(priority_badge(c["eff"], short=True), unsafe_allow_html=True)
            row[1].markdown(f"**{c['bed']}**")
            row[2].markdown(fmt_wait(c["waited"]))
            row[3].markdown(class_badge(c["request_class"]), unsafe_allow_html=True)
            row[4].markdown(c["transcript"] or "*(no transcript)*")
            flags = []
            if c["escalated"]:
                flags.append(f"⚠️ P{c['base_priority']}→P{c['final_priority']}: {c['reason']}")
            elif not c["speech_present"]:
                flags.append("No speech: routed like a call button")
            if c["bumps"]:
                flags.append(f"⏱️ waited too long: P{c['final_priority']}→P{c['eff']}")
            row[5].markdown("<br>".join(flags) or "—", unsafe_allow_html=True)
            if row[6].button("✓", key=f"ans_{c['id']}", help="Mark answered"):
                answer_call(c["id"])
                st.rerun(scope="fragment")

    queue_view()
    with st.expander("Audit log"):
        log = pd.read_sql("SELECT datetime(ts,'unixepoch','localtime') AS time, call_id, event, detail "
                          "FROM audit ORDER BY ts DESC LIMIT 200", db())
        st.dataframe(log, width="stretch", hide_index=True)
    if st.button("Clear queue and log"):
        db().execute("DELETE FROM calls"); db().execute("DELETE FROM audit"); db().commit()
        st.rerun()


def tab_submit(cached_mode: bool):
    bed = st.selectbox("Bed", CONFIG["beds"], index=13)
    demo = sorted(p for p in CONFIG["demo_dir"].glob("*") if p.suffix.lstrip(".").lower() in CONFIG["audio_types"])
    source = st.radio("Audio source", ["Demo clip", "Upload a file", "Record from mic"], horizontal=True,
                      help="File upload and demo clips are the reliable path; mic access depends on the browser.")
    name = data = None
    if source == "Demo clip":
        if not demo:
            st.warning(f"No clips in {CONFIG['demo_dir'].relative_to(ROOT)}.")
        else:
            pick = st.selectbox("Clip", [p.name for p in demo], key="demo_clip")
            name, data = pick, (CONFIG["demo_dir"] / pick).read_bytes()
    elif source == "Upload a file":
        up = st.file_uploader("Audio file", type=CONFIG["audio_types"])
        if up:
            name, data = up.name, up.getvalue()
    else:
        rec = st.audio_input("Record the request")
        if rec:
            name, data = "mic_recording.wav", rec.getvalue()

    if data:
        st.audio(data)
    if st.button("Submit call", type="primary", disabled=data is None):
        with st.spinner("Processing..."):
            try:
                r = run_call(name, data, bed, cached_mode)
            except Exception as e:  # keep the demo standing; point at cached mode
                st.error(f"Pipeline failed: {e}. Switch on cached mode in the sidebar.")
                r = None
        if r:
            add_call(bed, name, r)
            st.session_state["last_result"] = r
    if "last_result" in st.session_state:
        st.divider()
        st.markdown(f"#### Bed {st.session_state['last_result']['bed']} — "
                    f"{st.session_state['last_result']['audio_path']}")
        render_breakdown(st.session_state["last_result"])


def _read_json(name):
    try:
        return json.loads((CONFIG["results_dir"] / name).read_text())
    except FileNotFoundError:
        return None


def tab_results():
    res = CONFIG["results_dir"]
    st.markdown("Emergency recall is the headline metric, not accuracy (a missed emergency and a "
                "misrouted water request are not the same error).")

    tb = _read_json("text_branch_final.json")
    if tb:
        st.markdown("#### Text branch — Emergency recall")
        st.caption("Per recording (n = 6,661), 5-fold CV grouped by phrase.")
        c = st.columns(4)
        c[0].metric("ClinicalBERT, clean text", f"{tb['clinicalbert_clean']['emergency_recall']:.3f}",
                    help=f"{tb['clinicalbert_clean']['emergency_fn']} missed emergencies")
        c[1].metric("ClinicalBERT, on Whisper output", f"{tb['clinicalbert_on_asr']['emergency_recall']:.3f}",
                    help=f"{tb['clinicalbert_on_asr']['emergency_fn']} missed emergencies")
        st.caption(f"Missed emergencies: {tb['clinicalbert_clean']['emergency_fn']} on clean text, "
                   f"{tb['clinicalbert_on_asr']['emergency_fn']} on Whisper output.")
        c[2].metric("TF-IDF baseline", f"{tb['tfidf_phrase_level']['emergency_recall']:.3f}")
        c[3].metric("Whisper small WER", f"{tb['wer_full_corpus']:.1%}")

    for title, f in (("ClinicalBERT per-class", "clinicalbert_report.csv"), ("TF-IDF per-class", "tfidf_report.csv")):
        if (res / f).exists():
            st.markdown(f"**{title}** (recall is the column that matters; per unique phrase, n = 706)")
            st.dataframe(pd.read_csv(res / f, index_col=0).round(3), width="stretch")

    e2e = _read_json("end_to_end.json")
    if e2e:
        st.markdown("#### End to end")
        st.caption("Measured before the no-speech guard and before the notebook 10 refit.")
        c = st.columns(4)
        c[0].metric("Emergency reaching P1/P2", f"{e2e['emergency_reached_P1_P2']:.3f}")
        c[1].metric("Under-triaged", f"{e2e['under_triaged']:.3f}")
        c[2].metric("Over-triaged", f"{e2e['over_triaged']:.3f}")
        c[3].metric("Escalated by acoustics", e2e["escalated_by_acoustics"])

    gate = _read_json("distress_gate_results.json")
    if gate:
        st.markdown("#### Acoustic branch")
        st.warning("Under review: notebook 10 is testing how much of these numbers comes from "
                   "how the 4 s window was padded rather than from distress. Do not quote yet.")
        c = st.columns(3)
        c[0].metric("Distress AUC vs calm speech", gate["auc_vs_calm_speech"])
        c[1].metric("Distress AUC vs other vocalisations", gate["auc_vs_other_vocalisations"])
        op = gate["operating_point"]
        c[2].metric("Operating point", f"{op['recall']:.0%} recall", f"{op['false_alarm']:.1%} false alarms",
                    delta_color="off")

    figs = sorted(res.glob("**/*.png"))
    if figs:
        st.markdown("#### Figures")
        noise = [f for f in figs if "noise" in f.name]
        if not noise:
            st.info("Noise-degradation figure (notebook 07) not produced yet.")
        cols = st.columns(2)
        for i, f in enumerate(noise + [f for f in figs if f not in noise]):
            cols[i % 2].image(str(f), caption=str(f.relative_to(ROOT)), width="stretch")


# --------------------------------------------------------------------------- layout

with st.sidebar:
    st.markdown("## 🔔 Care-Voice AI")
    cached_mode = st.toggle("Cached mode", value=False,
                            help="Replay precomputed results instead of running the models live.")
    n_cached = len(load_cache())
    st.caption(f"{n_cached} clip(s) in cached_results.json" if n_cached else "No cached results yet.")
    st.caption("Escalation is an auditable rule (src/rules.py), not a trained fusion model.")

t1, t2, t3 = st.tabs(["Nurse Station", "Submit a Call", "Results"])
with t1:
    tab_nurse_station()
with t2:
    tab_submit(cached_mode)
with t3:
    tab_results()
