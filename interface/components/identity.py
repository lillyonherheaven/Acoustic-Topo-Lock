"""Product identity and lightweight HTML presentation primitives."""
from html import escape
from pathlib import Path

import streamlit as st

from acoustic_topo_lock.core.status import Check

ICONS = {
    "wave": '<path d="M2 12h3l2-7 4 14 3-11 2 4h6"/>',
    "shield": '<path d="M12 3 3 7v5c0 5 9 9 9 9s9-4 9-9V7z"/><path d="m8 12 3 3 5-6"/>',
    "room": '<path d="m3 7 9-4 9 4v11l-9 4-9-4zM3 7l9 4 9-4M12 11v11"/>',
    "replay": '<path d="M4 9a8 8 0 1 1 1 10M4 3v6h6"/><path d="m10 8 6 4-6 4z"/>',
    "chart": '<path d="M4 3v17h17M8 15l4-6 4 3 5-8"/>',
    "key": '<circle cx="8" cy="9" r="5"/><path d="m12 13 8 8m-4-4 3-3m-6 0 3-3"/>',
    "book": '<path d="M12 6c-4-3-8-2-10-1v15c3-2 7-2 10 0 3-2 7-2 10 0V5c-2-1-6-2-10 1v14"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 4 18 4 18 0V5M3 12c0 4 18 4 18 0"/>',
}


def icon(name, size=22):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{ICONS.get(name, ICONS["wave"])}</svg>')


def logo(size=46):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" aria-label="Acoustic Topo-Lock">
    <path d="M10 38V28a22 22 0 0 1 44 0v10M17 37V28a15 15 0 0 1 30 0v9M24 36v-8a8 8 0 0 1 16 0v8"
    stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <path d="M13 36h38v16a5 5 0 0 1-5 5H18a5 5 0 0 1-5-5z" stroke="currentColor" stroke-width="2"/>
    <circle cx="32" cy="43" r="3" fill="currentColor"/><path d="M32 45v6" stroke="currentColor" stroke-width="2"/>
    </svg>"""


def install_identity(root: Path):
    st.markdown("<style>" + (root / "interface/styles/workbench.css").read_text() + "</style>",
                unsafe_allow_html=True)
    if st.session_state.get("presentation", False):
        st.markdown('<style>.block-container{max-width:1500px!important}html{font-size:18px}'
                    '[data-testid="stSidebar"]{display:none}</style>', unsafe_allow_html=True)
        if st.button("Exit presentation", key="exit_presentation"):
            st.session_state.presentation = False
            st.rerun()


def brand():
    st.markdown(f'<div class="brand">{logo()}<div>TOPO<span>LOCK</span>'
                '<small>ACOUSTIC INTELLIGENCE</small></div></div>', unsafe_allow_html=True)


def page_header(title, subtitle, symbol="wave", eyebrow="ACOUSTIC TOPO-LOCK"):
    st.markdown(f'<div class="page-heading"><div class="eyebrow">{escape(eyebrow)}</div>'
                f'<h1>{icon(symbol,32)} {escape(title)}</h1><p>{escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def chips(items):
    markup = "".join(f'<span class="chip {"ready" if available else "muted"}">'
                     f'<i></i>{escape(label)}</span>' for label, available in items)
    st.markdown(f'<div class="chips">{markup}</div>', unsafe_allow_html=True)


def hero(service):
    records = [service.registry.inspect(check) for check in Check]
    bars = "".join(f'<span style="--h:{14 + ((i * 17 + i*i*3) % 70)}px;--d:{i * -0.07}s"></span>'
                   for i in range(61))
    st.markdown(f"""
    <section class="hero">
      <div class="eyebrow"><span class="signal-dot"></span> SPATIAL ACOUSTIC FINGERPRINTING</div>
      <div class="hero-grid"><div><h1>Every voice.<br>Every space.<br><em>A different signature.</em></h1>
      <p>Hear the identity. Read the room.<br>Examine the evidence behind every decision.</p></div>
      <div class="hero-art" aria-hidden="true"><div class="orb">{logo(150)}</div>
      <div class="waveform-art">{bars}</div><div class="art-caption">VOICE / SPACE / AUTHENTICITY</div></div></div>
    </section>""", unsafe_allow_html=True)
    chips([("Signal processing ready", True), *[
        (f'{record.check.value.title()} {"ready" if record.available else "unavailable"}', record.available)
        for record in records
    ]])


def section_card(title, body, symbol, footer):
    st.markdown(f'<div class="feature-card"><div class="card-icon">{icon(symbol,27)}</div>'
                f'<h3>{escape(title)}</h3><p>{escape(body)}</p><div class="card-footer">'
                f'{escape(footer)}</div></div>', unsafe_allow_html=True)