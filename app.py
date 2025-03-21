import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import polars as pl
import os
from sportsdataverse.nfl import load_nfl_pbp

# ✅ 2024 Opponent Schedule
KC_2024_SCHEDULE = {
    1: "BAL", 2: "CIN", 3: "ATL", 4: "LAC", 5: "NO",
    6: "BYE", 7: "SF", 8: "LV", 9: "TB", 10: "DEN", 11: "BUF",
    12: "CAR", 13: "LV", 14: "LAC", 15: "CLE", 16: "HOU", 17: "PIT", 18: "DEN"
}

# ✅ Optimized CSS for Light & Dark Mode
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Proxima Nova', sans-serif !important;
        }

        /* Sidebar styles */
        [data-testid="stSidebar"] {
            background-color: white;
            border-right: 1px solid #ddd;
        }

        /* Widget border fix */
        div[data-baseweb="select"] *, .stSlider > div > div {
            border-radius: 5px !important;
        }

        /* Text styling */
        h1, h2, h3, h4, h5, h6, p, .stText, .stTitle, .stSubheader {
            color: inherit !important;
        }

        /* Dark mode override */
        @media (prefers-color-scheme: dark) {
            html, body, [class*="css"] {
                background-color: #0e1117 !important;
                color: #e0e0e0 !important;
            }
            [data-testid="stSidebar"] {
                background-color: #0e1117 !important;
                border-right: 1px solid #333;
            }
            div[data-baseweb="select"] *, .stSlider > div > div {
                background-color: #1e1e1e !important;
                color: #e0e0e0 !important;
            }
        }

        /* Light mode override */
        @media (prefers-color-scheme: light) {
            html, body, [class*="css"] {
                background-color: white !important;
                color: #01284a !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Load and preprocess data
@st.cache_data
def load_data():
    raw_data = load_nfl_pbp(seasons=[2020, 2021, 2022, 2023, 2024])
    df = raw_data.to_pandas() if isinstance(raw_data, pl.DataFrame) else None
    if df is None:
        return None
    df = df[['season','week','qtr','game_seconds_remaining','down','ydstogo',
             'yardline_100','score_differential','play_type','shotgun','defteam','posteam']].dropna()
    df[['qtr','shotgun','game_seconds_remaining','ydstogo','yardline_100','score_differential']] = \
        df[['qtr','shotgun','game_seconds_remaining','ydstogo','yardline_100','score_differential']] \
        .apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    df['play_type_encoded'] = df['play_type'].apply(lambda x: 1 if x == "pass" else 0)
    df = df[df['posteam'] == "KC"]
    df_2024 = df[df['season'] == 2024]
    return pd.concat([df, df_2024, df_2024, df_2024], ignore_index=True)

df = load_data()

# ✅ Load or train model
def load_or_train_models(df):
    def train(df, shotgun, path):
        subset = df[df['shotgun'] == shotgun]
        X = subset[['qtr','game_seconds_remaining','down','ydstogo','yardline_100','score_differential']]
        y = subset['play_type_encoded']
        model = xgb.XGBClassifier(eval_metric='logloss')
        model.fit(X, y)
        model.save_model(path)
        return model

    if os.path.exists("model_shotgun.json") and os.path.exists("model_no_shotgun.json"):
        m1, m2 = xgb.XGBClassifier(), xgb.XGBClassifier()
        m1.load_model("model_shotgun.json")
        m2.load_model("model_no_shotgun.json")
    else:
        m1 = train(df, 1, "model_shotgun.json")
        m2 = train(df, 0, "model_no_shotgun.json")
    return {'shotgun': m1, 'no_shotgun': m2}

models = load_or_train_models(df)

# ✅ Sidebar Inputs
with st.sidebar:
    st.image("Eaglelogo2color.jpg", width=250)
    st.title("📊 PLAY PREDICTOR - KC CHIEFS")
    week = st.selectbox("SELECT GAME WEEK", list(KC_2024_SCHEDULE.keys()), index=7)
    opponent = KC_2024_SCHEDULE[week]
    if opponent != "BYE":
        st.markdown(f"### 🏈 OPPONENT: **{opponent}**")
    qtr = st.selectbox("SELECT QUARTER", [1, 2, 3, 4], index=2)
    minutes = st.selectbox("MINUTES REMAINING", list(range(15, -1, -1)), index=1)
    seconds = st.slider("SECONDS REMAINING", 0, 59, 14)
    down = st.selectbox("DOWN", [1, 2, 3, 4], index=0)
    ydstogo = st.slider("YARDS TO GO", 1, 30, 10)
    yardline = st.slider("FIELD POSITION", 1, 99, 20)
    score_diff = st.slider("SCORE DIFF (KC - OPP)", -30, 30, 4)
    go = st.button("🔍 GET PREDICTION")

# ✅ Prediction & Confidence
if go and opponent != "BYE":
    game_time = (minutes * 60) + seconds
    X_input = np.array([[qtr, game_time, down, ydstogo, yardline, score_diff]])
    shotgun = models['shotgun']
    no_shotgun = models['no_shotgun']

    def get_confidence(pred): return abs(pred - 50) * 2

    def similar_context(df, X_input):
        input_series = pd.Series(X_input[0], index=['qtr','game_seconds_remaining','down','ydstogo','yardline_100','score_differential'])
        diffs = df[['qtr','game_seconds_remaining','down','ydstogo','yardline_100','score_differential']].sub(input_series)
        distance = np.sqrt((diffs ** 2).sum(axis=1))
        similar = df[distance < 150]
        return len(similar), similar[['ydstogo', 'yardline_100', 'score_differential']].mean().to_dict()

    if shotgun and no_shotgun:
        ps = shotgun.predict_proba(X_input)[0][1] * 100
        rs = 100 - ps
        pn = no_shotgun.predict_proba(X_input)[0][1] * 100
        rn = 100 - pn

        conf_s = get_confidence(ps)
        conf_n = get_confidence(pn)

        count_s, avg_s = similar_context(df[df['shotgun'] == 1], X_input)
        count_n, avg_n = similar_context(df[df['shotgun'] == 0], X_input)

        st.subheader("🔮 PREDICTION RESULTS")
        st.write(f"📌 **WITH SHOTGUN:** {ps:.2f}% PASS / {rs:.2f}% RUN")
        st.write(f"✅ Confidence: {conf_s:.2f}% | 📚 Similar plays: {count_s}")
        st.write(f"📎 Context Avg — YTG: {avg_s['ydstogo']:.1f}, YL: {avg_s['yardline_100']:.1f}, SD: {avg_s['score_differential']:.1f}")

        st.write("---")

        st.write(f"📌 **WITHOUT SHOTGUN:** {pn:.2f}% PASS / {rn:.2f}% RUN")
        st.write(f"✅ Confidence: {conf_n:.2f}% | 📚 Similar plays: {count_n}")
        st.write(f"📎 Context Avg — YTG: {avg_n['ydstogo']:.1f}, YL: {avg_n['yardline_100']:.1f}, SD: {avg_n['score_differential']:.1f}")
    else:
        st.error("🚨 MODEL LOAD FAILED.")
