from html import escape
from math import cos, radians, sin

import plotly.graph_objects as go
import streamlit as st

from acoustic_topo_lock.core.status import Check, Decision, Mode, Status
from acoustic_topo_lock.preprocessing import AudioAnalysis
from interface.components.identity import icon

LABELS = {Check.VOICE: "Speaker verification", Check.ROOM: "Room compatibility", Check.REPLAY: "Replay detection"}
COLORS = {Decision.ACCEPT: "#6ce6d8", Decision.REJECT: "#f28b80", Decision.PARTIAL: "#e4b974",
          Decision.INSUFFICIENT: "#9eafc3", Decision.INVALID_AUDIO: "#f28b80"}
PASS = {Check.VOICE: Status.VERIFIED, Check.ROOM: Status.ROOM_MATCH, Check.REPLAY: Status.LIVE}
FAIL = {Check.VOICE: Status.NOT_VERIFIED, Check.ROOM: Status.ROOM_MISMATCH, Check.REPLAY: Status.REPLAY}


def model_status(service):
    return [{"check": LABELS[check], "available": record.available, "status": record.status,
             "model": record.name}
            for check in Check
            for record in (service.registry.inspect(check),)]


def signal_summary(audio: AudioAnalysis):
    metadata = audio.metadata
    columns = st.columns(4)
    columns[0].metric("Duration", f"{metadata.duration_seconds:.2f} s")
    columns[1].metric("Original sample rate", f"{metadata.sample_rate:,} Hz")
    columns[2].metric("Channels", metadata.channels)
    columns[3].metric("Detected activity", f"{audio.quality.active_seconds:.2f} s")
    st.caption(f"{metadata.filename}  ·  {metadata.format} / {metadata.subtype}  ·  Processed at "
               f"{audio.sample_rate:,} Hz")
    for warning in audio.warnings:
        st.warning(warning)


def score_gauge(item):
    if item.score is None:
        st.markdown('<div class="guidance">Score unavailable<br><small>'
                    'Awaiting measured model output</small></div>', unsafe_allow_html=True)
        return
    if item.check is Check.VOICE:
        low, high = -1.0, 1.0
    else:
        endpoints = [item.score, item.threshold] if item.threshold is not None else [item.score]
        spread = max(max(endpoints) - min(endpoints), abs(item.score) * .25, .1)
        low, high = min(endpoints) - spread, max(endpoints) + spread
    angle = -110 + 220 * (item.score - low) / (high - low)
    marker = ""
    if item.threshold is not None:
        theta = radians(-110 + 220 * (item.threshold - low) / (high - low) - 90)
        x1, y1 = 100 + 68 * cos(theta), 95 + 68 * sin(theta)
        x2, y2 = 100 + 81 * cos(theta), 95 + 81 * sin(theta)
        marker = f'<path d="M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}" stroke="#e4b974" stroke-width="3"/>'
    svg = f"""<svg class="gauge" viewBox="0 0 200 155" role="img" aria-label="Score {item.score:.4f}">
    <path d="M29.5 120.7 A75 75 0 1 1 170.5 120.7" fill="none" stroke="#29434a" stroke-width="5"/>
    {marker}<g class="needle" style="--angle:{angle:.2f}deg"><path d="M100 95V40" stroke="#6ce6d8" stroke-width="2"/>
    <circle cx="100" cy="95" r="4" fill="#6ce6d8"/></g>
    <text x="100" y="131" text-anchor="middle" fill="#ecf4f4" font-size="22">{item.score:.4f}</text>
    <text x="20" y="146" fill="#92a8b5" font-size="9">{low:.2f}</text>
    <text x="180" y="146" text-anchor="end" fill="#92a8b5" font-size="9">{high:.2f}</text></svg>"""
    st.markdown(svg, unsafe_allow_html=True)
    if item.threshold is not None:
        st.caption(f"Amber marker · threshold {item.threshold:.4f}")
    st.caption("Raw score · uncalibrated" if item.check is Check.VOICE else
               "Raw score · display range fitted to this result")


def evidence_radar(evidence):
    by_check = {item.check: item for item in evidence}
    figure = go.Figure()
    for check in Check:
        item = by_check.get(check)
        if item is not None and item.status is PASS[check]:
            radius, color, symbol, outcome = .85, "#6ce6d8", "circle", "Passed"
        elif item is not None and item.status is FAIL[check]:
            radius, color, symbol, outcome = .4, "#f28b80", "circle", "Failed"
        else:
            radius, color, symbol, outcome = 1.0, "#92a8b5", "circle-open", "Unavailable"
        figure.add_trace(go.Scatterpolar(
            r=[radius], theta=[check.value.title()], mode="markers",
            marker={"size":14, "symbol":symbol, "color":color, "line":{"width":2}},
            name=f"{check.value.title()} · {outcome}", hovertemplate=f"{check.value.title()}: {outcome}<extra></extra>",
        ))
    figure.update_layout(
        height=280, showlegend=False, margin={"l":65,"r":65,"t":25,"b":25},
        paper_bgcolor="rgba(0,0,0,0)", font={"color":"#adc3cd","family":"Barlow"},
        polar={"bgcolor":"rgba(0,0,0,0)", "radialaxis":{"range":[0,1.12],"visible":False},
               "angularaxis":{"rotation":90,"direction":"clockwise","gridcolor":"#29434a",
                              "categoryorder":"array","categoryarray":["Voice","Room","Replay"]}},
    )
    st.plotly_chart(figure, width="stretch")
    st.caption("Categorical evidence map · outer hollow rings = unavailable; "
               "cyan = passed; inner coral = failed. Position indicates status.")


def decision_summary(result):
    color = COLORS[result.decision.decision]
    st.markdown(f'<div class="final-banner" style="--state:{color}"><div class="eyebrow">'
                f'{escape(result.mode.value.upper())} / FINAL DECISION</div><h2>'
                f'{escape(result.decision.decision.value)}</h2><p>'
                f'{escape(result.decision.explanation[0])}</p></div>', unsafe_allow_html=True)
    renderer = {
        Decision.ACCEPT: st.success, Decision.REJECT: st.error, Decision.PARTIAL: st.warning,
        Decision.INSUFFICIENT: st.info, Decision.INVALID_AUDIO: st.error,
    }[result.decision.decision]
    renderer(f"**{result.decision.decision.value}** · {result.decision.completeness.replace('_', ' ').title()}")
    st.caption(f"{result.mode.value.title()} mode · {result.latency_ms:.1f} ms end to end")
    if result.evidence:
        for card, item in zip(st.columns(len(result.evidence)), result.evidence, strict=True):
            with card.container(border=True):
                symbol = {Check.VOICE:"wave", Check.ROOM:"room", Check.REPLAY:"replay"}[item.check]
                st.markdown(f'{icon(symbol)} **{LABELS[item.check]}**', unsafe_allow_html=True)
                st.markdown(f'<div class="evidence-status">{escape(item.status.value)}</div>',
                            unsafe_allow_html=True)
                score_gauge(item)
                st.write(item.reason)
                for key, value in item.details.items():
                    st.text(f"{key.replace('_', ' ').capitalize()} · {value}")
                if item.model_id:
                    st.caption(item.model_id if result.mode is Mode.DEMO else
                               f"{item.model_id} \u00b7 {item.latency_ms:.1f} ms")
                for warning in item.warnings:
                    st.caption(warning)
        with st.expander("Evidence contributions", expanded=True):
            evidence_radar(result.evidence)
    for warning in result.warnings:
        st.warning(warning)