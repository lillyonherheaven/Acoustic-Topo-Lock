from pathlib import Path

import streamlit as st

from acoustic_topo_lock.application import AuthenticationService
from acoustic_topo_lock.core.config_loader import load_settings
from acoustic_topo_lock.core.exceptions import ConfigError
from acoustic_topo_lock.core.logging_config import configure_logging
from interface.components.identity import brand, install_identity
from interface.pages.workbench import PAGES

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Acoustic Topo-Lock", page_icon="◉", layout="wide",
                   initial_sidebar_state="auto")

install_identity(ROOT)
try:
    settings = load_settings(root=ROOT)
except ConfigError as error:
    st.error(str(error))
    st.stop()

configure_logging(settings.log_level)
if "service" not in st.session_state or st.session_state.service.settings != settings:
    st.session_state.service = AuthenticationService(settings)
st.session_state.setdefault("speakers", {})
st.session_state.setdefault("rooms", {})
with st.sidebar:
    brand()
    page = st.radio("Workbench", list(PAGES), label_visibility="collapsed", key="navigation")
    st.divider()
    st.toggle("Presentation mode", key="presentation", help="Larger type and a full-width presentation canvas.")
    st.toggle("Guided demo", key="guided_demo", help="Step through labeled illustrative scenarios in Demo mode.")
    st.caption("ACOUSTIC TOPO-LOCK\nResearch workbench")
    st.caption("Local processing · Evidence-led decisions")

if st.session_state.get("presentation", False):
    page = st.selectbox("Presentation page", list(PAGES), index=list(PAGES).index(page),
                        key="presentation_page")

PAGES[page](st.session_state.service, ROOT)