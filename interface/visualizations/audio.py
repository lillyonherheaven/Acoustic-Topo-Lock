import numpy as np
import plotly.graph_objects as go
import streamlit as st

from acoustic_topo_lock.feature_extraction.mfcc_features import log_bands
from acoustic_topo_lock.feature_extraction.spectral_features import FFT_SIZE, spectrogram


def waveform(audio):
    samples = audio.waveform
    block = max(1, int(np.ceil(len(samples) / 2500)))
    starts = np.arange(0, len(samples), block)
    high = np.maximum.reduceat(samples, starts)
    low = np.minimum.reduceat(samples, starts)
    figure = go.Figure()
    time = starts / audio.sample_rate
    figure.add_scatter(x=time, y=low, mode="lines", line={"color": "#55daf4", "width": 0.7})
    figure.add_scatter(x=time, y=high, mode="lines", fill="tonexty",
                       line={"color": "#55daf4", "width": 0.7}, fillcolor="rgba(85,218,244,.25)")
    figure.update_layout(height=250, showlegend=False, xaxis_title="Time (s)",
                         yaxis_title="Amplitude", margin={"t": 12, "b": 12, "l": 12, "r": 12})
    _style(figure)
    st.plotly_chart(figure, width="stretch")


def spectrum(audio, *, mel: bool, key=None):
    times, frequencies, power, _ = spectrogram(audio.waveform, audio.sample_rate)
    step = max(1, len(times) // 500)
    power = power[::step]
    if mel:
        values = log_bands(power, audio.sample_rate, FFT_SIZE, mel=True) * (10 / np.log(10))
        frequencies = np.arange(1, values.shape[1] + 1)
    else:
        values = 10 * np.log10(np.maximum(power, 1e-12))
    values -= values.max()
    figure = go.Figure(go.Heatmap(x=times[::step], y=frequencies, z=values.T, zmin=-75, zmax=0,
                                  colorscale=[[0,"#0b1720"],[.5,"#24535b"],[1,"#8cebd7"]], colorbar={"title": "Relative dB"}))
    figure.update_layout(height=320, xaxis_title="Time (s)", yaxis_title="Mel band" if mel else
                         "Frequency (Hz)", margin={"t": 12, "b": 12, "l": 12, "r": 12})
    _style(figure)
    st.plotly_chart(figure, width="stretch", key=key)


def _style(figure):
    figure.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0c1b23",
                         font={"family":"Barlow","color":"#a9c1cd"})
    figure.update_xaxes(gridcolor="#20343c",zerolinecolor="#30434a")
    figure.update_yaxes(gridcolor="#20343c",zerolinecolor="#30434a")


def compare_waveforms(first, second):
    figure = go.Figure()
    for label, audio, color in (("A",first,"#6ce6d8"),("B",second,"#e4b974")):
        samples = audio.waveform
        block = max(1,int(np.ceil(len(samples)/2500)))
        starts = np.arange(0,len(samples),block)
        time = starts/audio.sample_rate
        high,low = np.maximum.reduceat(samples,starts),np.minimum.reduceat(samples,starts)
        figure.add_scatter(x=time,y=high,mode="lines",name=f"{label} peak envelope",
                           line={"color":color,"width":1})
        figure.add_scatter(x=time,y=low,mode="lines",name=label,showlegend=False,
                           line={"color":color,"width":1},opacity=.65)
    figure.update_layout(height=310,xaxis_title="Time (s)",yaxis_title="Amplitude",
                         margin={"t":25,"b":35,"l":45,"r":20},legend={"orientation":"h","y":1.15})
    _style(figure)
    st.plotly_chart(figure,width="stretch")
