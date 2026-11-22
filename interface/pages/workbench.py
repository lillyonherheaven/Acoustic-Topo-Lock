import json
from dataclasses import asdict

import plotly.graph_objects as go
import streamlit as st

from acoustic_topo_lock.application.demo import DEMO_CASES
from acoustic_topo_lock.application.result_formatter import result_dict
from acoustic_topo_lock.core.contracts import DEMO_WARNING, DEVELOPMENT_WARNING
from acoustic_topo_lock.core.exceptions import TopoLockError
from acoustic_topo_lock.core.status import Check, Mode
from acoustic_topo_lock.datasets.manifest import inspect_dataset, load_dataset
from acoustic_topo_lock.feature_extraction import extract_features, feature_names
from acoustic_topo_lock.utils.synthetic_audio import synthetic_wav
from interface.components.results import LABELS, decision_summary, model_status, signal_summary
from interface.visualizations.audio import spectrum, waveform, compare_waveforms
from interface.components.identity import chips, hero, page_header, section_card
from interface.components.live_scan import authenticate_with_scan, scan_html
from interface.components.research import research_metrics
from acoustic_topo_lock.core.status import Decision


def _upload(label, *, many=False):
    return st.file_uploader(label, type=["wav", "flac"], accept_multiple_files=many)


def _navigate(page):
    st.session_state.navigation = page


def home(service, _root):
    hero(service)
    left, right = st.columns([1, 1])
    left.button("Start authentication", type="primary", on_click=_navigate, args=("Authentication",))
    right.button("Explore acoustic signals", on_click=_navigate, args=("Audio Analysis",))
    st.write("")
    cards = (
        ("Voice identity", "Compare a consented voice reference with the speaker in a recording.", "wave", "ECAPA · SPEAKER EMBEDDINGS"),
        ("Room signature", "Examine the acoustic context surrounding the voice.", "room", "SPECTRAL + DECAY FEATURES"),
        ("Replay evidence", "Evaluate traces introduced by recording and playback.", "replay", "INDEPENDENT REPLAY CHECK"),
    )
    for column, values in zip(st.columns(3), cards, strict=True):
        with column:
            section_card(*values)
    st.write("")
    with st.container(border=True):
        st.subheader("Follow the evidence")
        st.write("Voice, room, and replay checks contribute independently. Open a scan to inspect the result, "
                 "its measured timing, and any missing evidence.")
        st.caption("Demo scenarios are clearly labeled. Real mode uses available local models.")


def audio_analysis(service, _root):
    page_header("The shape of a signal", "Inspect a recording or compare two acoustic signatures.", "wave",
                "SIGNAL LAB / AUDIO ANALYSIS")
    compare = st.toggle("A/B comparison", key="audio_compare")
    if compare:
        columns = st.columns(2)
        with columns[0]:
            first = st.file_uploader("Recording A", type=["wav","flac"], key="compare_a")
        with columns[1]:
            second = st.file_uploader("Recording B", type=["wav","flac"], key="compare_b")
        if first is None or second is None:
            st.info("Upload two recordings to compare waveforms, spectrograms, and measured levels.")
            return
        try:
            analyses = [service.analyze(item.getvalue(), filename=item.name) for item in (first,second)]
        except TopoLockError as error:
            st.error(str(error))
            return
        compare_waveforms(*analyses)
        mel = st.toggle("Show Mel bands", key="compare_mel")
        for column, label, audio, item in zip(st.columns(2), ("A","B"), analyses, (first,second), strict=True):
            with column:
                st.subheader(f"Recording {label}")
                st.audio(item.getvalue())
                spectrum(audio, mel=mel, key=f"spectrum_{label}")
                st.metric("RMS amplitude", f"{audio.quality.rms:.6f}")
                st.metric("Peak amplitude", f"{audio.quality.peak:.6f}")
                st.caption(f"{audio.metadata.duration_seconds:.2f} s · {audio.metadata.sample_rate:,} Hz source")
                for warning in audio.warnings:
                    st.warning(warning)
        return
    use_fixture = st.toggle("Use generated test audio", value=True)
    uploaded = None if use_fixture else _upload("Upload a recording")
    if not use_fixture and uploaded is None:
        return
    if use_fixture:
        st.info(DEVELOPMENT_WARNING)
    raw, name = (synthetic_wav(), "synthetic.wav") if use_fixture else (uploaded.getvalue(), uploaded.name)
    try:
        audio = service.analyze(raw, filename=name)
    except TopoLockError as error:
        st.error(str(error))
        return
    signal_summary(audio)
    st.audio(raw, format="audio/wav" if audio.metadata.format != "FLAC" else "audio/flac")
    st.subheader("Waveform")
    waveform(audio)
    st.subheader("Spectral signature")
    mel = st.toggle("Show Mel bands")
    spectrum(audio, mel=mel)
    with st.expander("Levels and DSP features"):
        st.json(asdict(audio.quality), expanded=False)
        kind = st.selectbox("Feature set", ["room", "replay", "voice"])
        options = {"include_lfcc": service.settings.include_lfcc}
        values = extract_features(audio, kind, **options)
        st.dataframe([{"feature": key, "value": value} for key, value in
                      zip(feature_names(kind, **options), values, strict=True)], hide_index=True)


def enrollment(service, _root):
    page_header("Build a reference", "Enroll consented recordings for speaker and room comparison.", "key", "IDENTITY / ENROLLMENT")
    st.write("Profiles are held in this browser session. Use recordings shared with the speaker’s consent.")
    kind = st.radio("Enrollment type", ["Speaker", "Room"], horizontal=True)
    states = st.session_state.speakers if kind == "Speaker" else st.session_state.rooms
    if kind == "Speaker":
        record = service.registry.inspect(Check.VOICE)
        st.caption(record.status)
    else:
        st.caption("Room profiles summarize reference recordings. Inference requires an installed classifier "
                   "trained on the room label.")
    with st.form(f"enroll_{kind}", clear_on_submit=True):
        identifier = st.text_input(f"{kind} ID")
        sources = _upload("Reference recordings", many=True)
        consent = st.checkbox("All recordings were shared with consent.")
        submitted = st.form_submit_button(f"Save {kind.lower()} profile")
    if submitted:
        if not sources:
            st.error("Provide reference recordings.")
        else:
            enroll = service.enroll_voice if kind == "Speaker" else service.enroll_room
            try:
                profile = enroll(identifier, ((item.getvalue(), item.name) for item in sources),
                                 consented=consent)
            except TopoLockError as error:
                st.error(str(error))
            else:
                states[profile.identifier] = profile
                st.success(f"Saved {profile.identifier} from {profile.sample_count} reference recordings.")
                st.toast("Reference profile saved")
    if states:
        st.subheader("Session profiles")
        selected = st.selectbox("Profile", list(states), key=f"remove_{kind}")
        if st.button("Remove selected profile", key=f"delete_{kind}"):
            del states[selected]
            st.rerun()


def _show_scan_result(result):
    decision_summary(result)
    st.download_button("Download evidence report", json.dumps(result_dict(result), indent=2),
                       file_name="verification-result.json", mime="application/json", on_click="ignore")


def authentication(service, _root):
    page_header("Read the evidence", "One recording. Three independent checks. An explainable decision.",
                "shield", "LIVE SCAN / AUTHENTICATION")
    mode = st.radio("Mode", list(Mode), index=1, horizontal=True, format_func=lambda item: item.value.title())
    if mode is Mode.DEMO:
        st.warning(DEMO_WARNING)
        guided = st.session_state.get("guided_demo",False)
        cases = list(DEMO_CASES)
        if guided:
            st.session_state.setdefault("guided_step",0)
            index = st.session_state.guided_step % len(cases)
            case = cases[index]
            st.subheader(f"Guided demo · {index + 1} of {len(cases)}")
            st.write(case)
            st.caption("Run this illustrative scenario, inspect its evidence, then continue.")
        else:
            case = st.selectbox("Illustrative scenario", cases, index=1)
        key = ("demo",case)
        if st.button("Run demo", type="primary"):
            result = authenticate_with_scan(service,synthetic_wav(),filename="synthetic.wav",mode=mode,demo_case=case)
            st.session_state.scan_result = (key,result,st.session_state.last_scan)
            st.toast("Demo evidence ready")
        if guided and st.button("Next scenario"):
            st.session_state.guided_step = (index + 1) % len(cases)
            st.rerun()
    elif mode is Mode.DEVELOPMENT:
        st.info(DEVELOPMENT_WARNING)
        uploaded = _upload("Upload a test recording")
        raw, name = (uploaded.getvalue(), uploaded.name) if uploaded else (synthetic_wav(), "synthetic.wav")
        key = ("development",name,raw)
        if st.button("Run development pipeline", type="primary"):
            result = authenticate_with_scan(service,raw,filename=name,mode=mode)
            st.session_state.scan_result = (key,result,st.session_state.last_scan)
    else:
        speakers = st.session_state.speakers
        enrolled = st.selectbox("Speaker profile", [None,*speakers],format_func=lambda item:item or "Select a profile")
        rooms = st.session_state.rooms
        room_label = st.selectbox("Room profile",[None,*rooms],format_func=lambda item:item or "Enter a room label")
        if room_label is None:
            room_label = st.text_input("Authorized room label")
        uploaded = _upload("Probe recording")
        raw = uploaded.getvalue() if uploaded else None
        key = ("real",enrolled,room_label,raw)
        if st.button("Verify recording", type="primary", disabled=uploaded is None):
            result = authenticate_with_scan(service,raw,speakers.get(enrolled),room_label.strip() or None,
                                            mode=mode,filename=uploaded.name)
            st.session_state.scan_result = (key,result,st.session_state.last_scan)
            st.toast("Evidence ready")
    saved = st.session_state.get("scan_result")
    if saved and saved[0] == key:
        st.markdown(scan_html(saved[2]),unsafe_allow_html=True)
        _show_scan_result(saved[1])
    else:
        st.markdown(scan_html({}),unsafe_allow_html=True)
        st.caption("Stages update as service calls complete. A result appears after the pipeline returns.")


def models(service, _root):
    page_header("System readiness", "Current model availability from the local registry.", "shield", "OPERATIONS / MODELS")
    st.caption("Availability here checks local files and dependencies. The inference services validate artifacts when loaded.")
    for check in Check:
        record = service.registry.inspect(check)
        with st.container(border=True):
            st.subheader(LABELS[check])
            st.write(record.status)
            st.caption(str(record.path))
            if record.metadata:
                with st.expander("Artifact metadata"):
                    st.json(record.metadata)


def datasets(service, _root):
    page_header("Data provenance", "Inspect the manifests behind each experiment.", "database", "RESEARCH / DATASETS")
    st.write("Inspect local manifest references, labels, and participant counts before training.")
    kind = st.radio("Dataset use", ["replay", "room"], horizontal=True)
    path = service.settings.data_root / "metadata" / f"{kind}.json"
    st.caption(str(path))
    if not path.is_file():
        st.info("No dataset manifest has been installed for this module.")
        return
    if st.button("Inspect dataset"):
        try:
            st.json(inspect_dataset(load_dataset(path, kind)))
        except TopoLockError as error:
            st.error(str(error))



def documentation(_service, root):
    page_header("Inside Topo-Lock", "Architecture, research methodology, and operating guides.", "book", "KNOWLEDGE / DOCUMENTATION")
    files = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    selection = st.selectbox("Guide", files, format_func=lambda path: path.stem.replace("_", " ").title())
    try:
        st.markdown(selection.read_text(encoding="utf-8"))
    except OSError:
        st.warning("This guide is missing from the current checkout.")


PAGES = {
    "Home Dashboard": home,
    "Audio Analysis": audio_analysis,
    "Voice Enrollment": enrollment,
    "Authentication": authentication,
    "Model Status": models,
    "Dataset Information": datasets,
    "Research Metrics": research_metrics,
    "System Documentation": documentation,
}
