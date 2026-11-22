"""Observe existing bound service calls without altering the backend pipeline."""
from copy import copy
from html import escape
from time import perf_counter
from types import SimpleNamespace

import streamlit as st

from acoustic_topo_lock.core.status import Mode, Status
from acoustic_topo_lock.core.exceptions import AudioError

STAGES = ("Audio", "Voice", "Room", "Replay", "Decision")
UNAVAILABLE = {Status.MODEL_UNAVAILABLE, Status.ROOM_MODEL_UNAVAILABLE, Status.REPLAY_MODEL_NOT_TRAINED,
               Status.ENROLLMENT_REQUIRED, Status.ROOM_ENROLLMENT_REQUIRED, Status.MODULE_ERROR}


def scan_html(events, running=None):
    nodes = []
    for name in STAGES:
        if name in events:
            milliseconds, status = events[name]
            state = "unavailable" if status in UNAVAILABLE else "complete"
            detail = str(status.value) if isinstance(status, Status) else str(status)
            timing = f"{milliseconds:.1f} ms" if milliseconds is not None else "No model inference"
        else:
            state = "running" if running == name else ""
            detail = "Processing" if running == name else "Awaiting evidence"
            timing = "—"
        nodes.append(f'<div class="scan-node {state}"><strong>{name}</strong><small>{timing}</small>'
                     f'<small>{escape(detail.replace("_", " "))}</small></div>')
    return '<div class="scan">' + "".join(nodes) + "</div>"


def authenticate_with_scan(service, *args, **kwargs):
    """Show completions as the existing service returns each real stage."""
    placeholder = st.empty()
    events = {}
    observed = copy(service)

    def render(running=None):
        placeholder.markdown(scan_html(events, running), unsafe_allow_html=True)

    def wrap(name, function):
        def measured(*values, **options):
            render(name)
            start = perf_counter()
            try:
                result = function(*values, **options)
            except Exception as error:
                status = Status.INVALID_AUDIO if isinstance(error, AudioError) else Status.MODULE_ERROR
                events[name] = ((perf_counter() - start) * 1000, status)
                render()
                raise
            status = getattr(result, "status", "Processed")
            events[name] = ((perf_counter() - start) * 1000, status)
            render()
            return result
        return measured

    observed.analyze = wrap("Audio", service.analyze)
    for name in ("voice", "room", "replay"):
        original = getattr(service, name)
        setattr(observed, name, SimpleNamespace(verify=wrap(name.title(), original.verify)))
    render("Audio")
    result = observed.authenticate_audio(*args, **kwargs)
    # The backend's end-to-end latency includes fusion. Do not fabricate a separate fusion time.
    if result.mode is Mode.DEMO:
        for item in result.evidence:
            events[item.check.value.title()] = (None, f"Illustrative / {item.status.value}")
    events["Decision"] = (result.latency_ms, "Total elapsed")
    render()
    if result.mode is Mode.DEMO:
        st.caption("Demo check outcomes are illustrative. Only audio processing and total elapsed time were measured.")
    st.session_state["last_scan"] = events
    placeholder.empty()
    return result