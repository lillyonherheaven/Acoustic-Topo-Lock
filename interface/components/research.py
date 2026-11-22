"""Render saved reports and assets. All measured values come from supplied JSON."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from interface.components.identity import chips, page_header


def load_report(data: bytes) -> dict:
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("Metrics JSON exceeds the 10 MB limit.")
    report = json.loads(data)
    if not isinstance(report, dict):
        raise ValueError("Expected a JSON report object.")
    return report


def confusion_figure(metrics):
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    labels = metrics.get("confusion_matrix_labels", metrics.get("labels"))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("Confusion matrix must be square.")
    if not np.isfinite(matrix).all() or (matrix < 0).any() or not np.equal(matrix, np.floor(matrix)).all():
        raise ValueError("Confusion matrix must contain finite nonnegative counts.")
    if labels is None or len(labels) != len(matrix) or not all(isinstance(label,str) for label in labels):
        raise ValueError("The report must supply the confusion matrix label order.")
    figure = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, text=matrix.astype(np.int64),
        texttemplate="%{text:,}", colorscale=[[0,"#112932"],[1,"#6ce6d8"]],
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z:,} trials<extra></extra>",
        colorbar={"title":"Trials"}, textfont={"size":24},
    ))
    figure.update_layout(height=380, xaxis_title="Predicted class", yaxis_title="Actual class",
                         yaxis={"autorange":"reversed"}, margin={"l":65,"r":20,"t":20,"b":55},
                         paper_bgcolor="rgba(0,0,0,0)", font={"color":"#b5c9d1"})
    return figure


def local_reports(root):
    directories = [root / "results/reports", root / "replay_detection/results"]
    reports = set()
    for directory in directories:
        reports.update(directory.rglob("metrics.json"))
        reports.update(directory.rglob("*-eval.json"))
    return sorted(reports)


def _display_metrics(metrics):
    names = (("eer","EER"),("far","Replay acceptance / FAR"),("frr","Live rejection / FRR"),
             ("roc_auc","ROC AUC"),("accuracy","Accuracy"))
    present = [(key,label) for key,label in names if key in metrics]
    for columns_start in range(0, len(present), 3):
        row = present[columns_start:columns_start+3]
        for column, (key,label) in zip(st.columns(len(row)), row, strict=True):
            value = metrics[key]
            if type(value) not in (int,float) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {key} in the report.")
            column.metric(label, f"{value:.2%}" if key != "roc_auc" else f"{value:.4f}")


def _asset(source, uploads, filename):
    if filename in uploads:
        return uploads[filename]
    if source is not None:
        path = source.parent / filename
        if path.is_file() and path.stat().st_size <= 15 * 1024 * 1024:
            return path.read_bytes()
    return None


def _diagnostic(source, uploads, split, kind):
    asset = _asset(source, uploads, f"{split}_{kind}.png")
    if asset is None:
        st.info(f"{kind.upper()} diagnostic unavailable. Add {split}_{kind}.png from this run.")
    else:
        image = Image.open(io.BytesIO(asset))
        if image.width * image.height > 25_000_000:
            raise ValueError("Diagnostic image dimensions exceed the display limit.")
        image.load()
        st.image(image, caption=f"{split.upper()} · saved {kind.upper()} diagnostic", width="stretch")


def research_metrics(_service, root):
    page_header("Measured performance", "Trace every result to its saved evaluation report.",
                "chart", "RESEARCH / REPLAY DETECTION")
    files = local_reports(root)
    with st.expander("Load evaluation outputs", expanded=not files):
        uploaded = st.file_uploader("metrics.json and matching diagnostic PNGs", type=["json","png"],
                                    accept_multiple_files=True, key="research_assets")
        st.caption("Load the full-run JSON with eval_confusion.png, eval_roc.png and eval_det.png. "
                   "Local reports are also discovered under results/reports and replay_detection/results.")
    json_uploads = {item.name:item for item in uploaded if item.name.endswith(".json")}
    images = {item.name:item.getvalue() for item in uploaded if item.name.endswith(".png")}
    choices = [*files, *json_uploads]
    if not choices:
        chips([("Evaluation outputs unavailable",False)])
        st.markdown('<div class="guidance">Ready for your measured run.<br>'
                    'The notebook and source package are present. Supply metrics.json and its diagnostic PNGs '
                    'to populate this page.</div>', unsafe_allow_html=True)
        st.caption("No performance values are inferred from notebook output.")
        return
    choice = st.selectbox("Evaluation report", choices,
                          format_func=lambda value: value.name if isinstance(value,Path) else value)
    source = choice if isinstance(choice,Path) else None
    try:
        data = source.read_bytes() if source else json_uploads[choice].getvalue()
        report = load_report(data)
        full_format = "dev" in report or "eval" in report
        if full_format:
            if report.get("mode") != "full":
                st.warning("This report is not marked as a full run. Review the raw provenance below.")
                with st.expander("Source report",expanded=True):
                    st.json(report)
                return
            options = [split for split in ("eval","dev") if isinstance(report.get(split),dict)
                       and (report[split].get("status") == "measured" or
                            (split == "dev" and "confusion_matrix" in report[split]))]
            if not options:
                st.info("No measured partition is available in this report.")
                st.json(report)
                return
            split = st.radio("Measured split",options,horizontal=True)
            metrics = report[split]
        else:
            split = report.get("partition","unspecified")
            metrics = report.get("metrics",{})
        dataset = report.get("dataset",{})
        dataset_name = (dataset.get("name") or dataset.get("dataset")) if isinstance(dataset,dict) else str(dataset)
        dev_threshold = report.get("dev",{}).get("threshold")
        threshold = metrics.get("threshold", metrics.get("threshold_from_development"))
        selection = report.get("selection","")
        frozen = (dev_threshold is not None and threshold == dev_threshold and
                  "development" in str(selection).lower())
        chips([(dataset_name or "Dataset · see source provenance", bool(dataset_name)),
               (f"Split · {str(split).upper()}", True),
               ("Dev threshold · frozen" if frozen else "Threshold provenance · see report",frozen),
               ("Full run" if report.get("mode") == "full" else "Saved evaluation",True)])
        st.caption(f"Source · {choice.name if source else choice}  /  SHA-256 {hashlib.sha256(data).hexdigest()[:16]}")
        if "selected_model" in report:
            st.subheader(str(report["selected_model"]).replace("_"," ").title())
        _display_metrics(metrics)
        if threshold is not None:
            st.caption(f"Operating threshold · {threshold} · raw model score")
        tabs = st.tabs(["Confusion matrix","ROC","DET","Provenance"])
        with tabs[0]:
            if "confusion_matrix" in metrics:
                st.plotly_chart(confusion_figure(metrics), width="stretch")
                st.caption("Counts and class order are loaded from the selected JSON.")
            else:
                st.info("Confusion matrix counts are absent from this JSON.")
            saved = _asset(source,images,f"{split}_confusion.png")
            if saved:
                with st.expander("Original exported confusion image"):
                    _diagnostic(source,images,split,"confusion")
        with tabs[1]:
            curve = report.get("curve",{})
            if "fpr" in curve and "tpr" in curve:
                figure = go.Figure(go.Scatter(x=curve["fpr"],y=curve["tpr"],mode="lines"))
                figure.update_layout(xaxis_title="False positive rate",yaxis_title="True positive rate")
                st.plotly_chart(figure,width="stretch")
            else:
                _diagnostic(source,images,split,"roc")
        with tabs[2]:
            curve = report.get("curve",{})
            if "far" in curve and "frr" in curve:
                figure = go.Figure(go.Scatter(x=curve["far"],y=curve["frr"],mode="lines"))
                figure.update_layout(xaxis_title="False acceptance rate",yaxis_title="False rejection rate")
                st.plotly_chart(figure,width="stretch")
            else:
                _diagnostic(source,images,split,"det")
        with tabs[3]:
            st.json(report,expanded=False)
            st.download_button("Download source JSON",data,file_name="metrics.json",mime="application/json")
            latency = report.get("inference_latency")
            if latency:
                st.write("Inference latency from the saved run")
                st.json(latency,expanded=False)
    except (OSError,ValueError,KeyError,TypeError,OverflowError) as error:
        st.error(f"Could not display these outputs: {error}")