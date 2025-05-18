# -*- coding: utf-8 -*-
"""
Streamlit PV Dashboard cu upload modele
"""

import pandas as pd
import numpy as np
import requests
import joblib
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz

# ─────────────
# A) Configurații generale
# ─────────────
lat, lon = 44.4268, 26.1025
tz = pytz.timezone("Europe/Bucharest")
API_KEY = "d2af965998c24aeb88395350251104"

# ─────────────
# Upload modele din UI
# ─────────────
st.sidebar.subheader("🔁 Încarcă Modele & Scaler")

gru_file = st.sidebar.file_uploader("Model GRU (.keras)", type=["keras"])
lstm12_file = st.sidebar.file_uploader("Model LSTM 12-36 (.keras)", type=["keras"])
lstm7d_file = st.sidebar.file_uploader("Model LSTM 7d (.keras)", type=["keras"])
scaler_file = st.sidebar.file_uploader("Scaler (.pkl)", type=["pkl"])

mdl_12h = mdl_12_36h = mdl_7d = scaler = None

if gru_file:
    mdl_12h = tf.keras.models.load_model(gru_file)
    st.sidebar.success("GRU încărcat!")
if lstm12_file:
    mdl_12_36h = tf.keras.models.load_model(lstm12_file)
    st.sidebar.success("LSTM 12-36 încărcat!")
if lstm7d_file:
    mdl_7d = tf.keras.models.load_model(lstm7d_file)
    st.sidebar.success("LSTM 7d încărcat!")
if scaler_file:
    scaler = joblib.load(scaler_file)
    st.sidebar.success("Scaler încărcat!")

# ─────────────
# Funcție meteo
# ─────────────
@st.cache_data
def fetch_current_weather():
    r = requests.get(
        "http://api.weatherapi.com/v1/current.json",
        params={"key": API_KEY, "q": f"{lat},{lon}", "aqi": "no"}
    ).json()['current']
    return {
        "icon":     "https:" + r['condition']['icon'],
        "text":     r['condition']['text'],
        "temp_c":   r['temp_c'],
        "humidity": r['humidity'],
        "wind_kph": r['wind_kph'],
        "pressure": r['pressure_mb'],
        "uv":       r['uv']
    }

@st.cache_data
def load_calendar(threshold):
    t0 = datetime.now(tz)
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=shortwave_radiation"
        "&forecast_days=7"
        "&timezone=Europe/Bucharest"
    )
    H = requests.get(url).json()['hourly']
    df = pd.DataFrame({'sr': H['shortwave_radiation']})
    df.index = pd.DatetimeIndex(pd.to_datetime(H['time'])).tz_localize(tz)
    df = df.loc[t0 : t0 + timedelta(days=7)]
    df['will_charge'] = (df['sr'] > threshold).astype(int)
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    return df

def find_blocks(df, min_len=2):
    blocks = {}
    for date, sub in df.groupby('date'):
        bl, cnt, start, prev = [], 0, None, None
        for ts, r in sub.iterrows():
            if r['will_charge'] == 1:
                if cnt == 0: start = ts
                cnt += 1
            else:
                if cnt >= min_len: bl.append((start, prev))
                cnt = 0
            prev = ts
        if cnt >= min_len: bl.append((start, prev))
        blocks[date] = bl
    return blocks

# ─────────────
# UI Principal
# ─────────────
st.set_page_config(page_title="Dashboard PV", layout="wide")
st.title("Dashboard PV – 7 zile")

st.sidebar.header("Setări PV")
threshold = st.sidebar.slider("Prag radiație (W/m²)", 100, 1000, 200, 10)
area      = st.sidebar.number_input("Suprafață panouri (m²)", 1.0, 10000.0, 100.0, 1.0)
eff_pct   = st.sidebar.slider("Randament (%)", 5, 25, 15) / 100.0

with st.expander("🌤️ Condiții meteo curente", expanded=True):
    cw = fetch_current_weather()
    c1, c2, c3, c4 = st.columns(4)
    c1.image(cw["icon"], width=64); c1.write(f"**{cw['text']}**")
    c2.metric("Temperatură", f"{cw['temp_c']} °C"); c2.metric("Umiditate", f"{cw['humidity']} %")
    c3.metric("Vânt", f"{cw['wind_kph']:.1f} km/h"); c3.metric("UV Index", cw["uv"])
    c4.metric("Presiune", f"{cw['pressure']} mb")

df = load_calendar(threshold)
blocks = find_blocks(df)

records = []
for date, bl in blocks.items():
    for s, e in bl:
        idx = pd.date_range(s, e, freq='h')
        kwh = (df.loc[idx, 'sr'] * area * eff_pct).sum() / 1000
        records.append({
            "Data": date,
            "Ore recomandate de incarcare": f"{s:%H:%M}–{e:%H:%M}",
            "kWh estimați": kwh
        })
df_records = pd.DataFrame(records)

styled = (
    df_records.style
    .format({"kWh estimați": "{:.2f}"})
    .set_table_styles([
        {"selector": "th", "props": [("font-size", "18px"), ("font-weight", "bold")]},
        {"selector": "td", "props": [("font-size", "16px")]}
    ])
    .set_table_attributes('style="width:100%"')
)
st.subheader("Estimare producție PV pe blocuri")
st.write(styled.to_html(), unsafe_allow_html=True)

pivot = df.pivot(index='date', columns='hour', values='will_charge').fillna(0)
dates = pivot.index.astype(str); hours = pivot.columns.tolist()

fig, ax = plt.subplots(figsize=(12, len(dates)*0.5))
cax = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="Greens", vmin=0, vmax=1)
ax.set_yticks(np.arange(len(dates))); ax.set_yticklabels(dates)
ax.set_xticks(np.arange(len(hours))); ax.set_xticklabels(hours)
ax.set_xlabel("Ora"); ax.set_ylabel("Data")
fig.colorbar(cax, ax=ax, ticks=[0,1]).ax.set_yticklabels(["0 = fără încărcare", "1 = încărcare"])

for i, date in enumerate(pivot.index):
    for s, _ in blocks[date]:
        ax.scatter(s.hour, i, s=200, facecolors="none", edgecolors="red", linewidth=2)

st.pyplot(fig)
