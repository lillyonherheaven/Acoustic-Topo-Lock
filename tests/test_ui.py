import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest


def test_every_page_and_demo():
    app = AppTest.from_file("app.py").run(timeout=20)
    assert not app.exception
    pages = app.sidebar.radio[0].options
    for page in pages:
        app.sidebar.radio[0].set_value(page).run(timeout=20)
        assert not app.exception, page
        if page == "Authentication":
            app.button[0].click().run(timeout=20)
            assert not app.exception
            assert "REJECT" in app.error[0].value
            assert any("DEMO DATA" in warning.value for warning in app.warning)


def test_metrics_counts_and_validation():
    from interface.components.research import confusion_figure, load_report
    report = load_report(b'{"mode":"full","eval":{"confusion_matrix":[[3,1],[2,4]],"confusion_matrix_labels":["live","replay"]}}')
    figure = confusion_figure(report["eval"])
    assert figure.data[0].z.tolist() == [[3,1],[2,4]]
    with pytest.raises(ValueError, match="nonnegative counts"):
        confusion_figure({"confusion_matrix":[[1,-1],[0,1]],"confusion_matrix_labels":["live","replay"]})
    with pytest.raises(ValueError, match="label order"):
        confusion_figure({"confusion_matrix":[[1,0],[0,1]]})


def test_live_scan_observes_missing_modules(tmp_path):
    from acoustic_topo_lock.application import AuthenticationService
    from acoustic_topo_lock.core.config_loader import load_settings
    from acoustic_topo_lock.core.status import Decision, Status
    from acoustic_topo_lock.utils.synthetic_audio import synthetic_wav
    from interface.components.live_scan import authenticate_with_scan
    import streamlit as st

    service = AuthenticationService(load_settings(root=tmp_path))
    result = authenticate_with_scan(service, synthetic_wav(), filename="software-fixture.wav", mode="real")
    assert result.decision.decision is Decision.INSUFFICIENT
    events = st.session_state.last_scan
    assert set(events) == {"Audio", "Voice", "Room", "Replay", "Decision"}
    assert events["Replay"][1] is Status.REPLAY_MODEL_NOT_TRAINED
    assert all(milliseconds >= 0 for milliseconds, _ in events.values())


def test_demo_results_do_not_cross_modes():
    app = AppTest.from_file("app.py").run(timeout=20)
    app.sidebar.radio[0].set_value("Authentication").run(timeout=20)
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert any("REJECT" in error.value for error in app.error)
    from acoustic_topo_lock.core.status import Mode
    app.main.radio[0].set_value(Mode.REAL).run(timeout=20)
    assert not app.exception
    assert not any("REJECT" in error.value for error in app.error)
