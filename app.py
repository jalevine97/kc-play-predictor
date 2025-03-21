
import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import polars as pl
import os
import hashlib
from sportsdataverse.nfl import load_nfl_pbp

# ✅ Optimized CSS for Light & Dark Mode
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Proxima Nova', sans-serif !important;
        }
        [data-testid="stSidebar"] {
            background-color: white;
            border-right: 1px solid #ddd;
        }
        div[data-baseweb="select"] *, .stSlider > div > div {
            border-radius: 5px !important;
        }
        h1, h2, h3, h4, h5, h6, p, .stText, .stTitle, .stSubheader {
            color: inherit !important;
        }
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
        @media (prefers-color-scheme: light) {
            html, body, [class*="css"] {
                background-color: white !important;
                color: #01284a !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ✅ 2024 Opponent Schedule
KC_2024_SCHEDULE = {
    1: "BAL", 2: "CIN", 3: "ATL", 4: "LAC", 5: "NO",
    6: "BYE", 7: "SF", 8: "LV", 9: "TB", 10: "DEN", 11: "BUF",
    12: "CAR", 13: "LV", 14: "LAC", 15: "CLE", 16: "HOU", 17: "PIT", 18: "DEN"
}

@st.cache_data
def load_data():
    raw_data = load_nfl_pbp(seasons=[2020, 2021, 2022, 2023, 2024])
    df = raw_data.to_pandas() if isinstance(raw_data, pl.DataFrame) else None
    if df is None:
        return None
    df = df[['season','week','qtr','game_seconds_remaining','down','ydstogo',
             'yardline_100','score_differential','play_type','shotgun','defteam','posteam']].dropna()
    df[['qtr','shotgun','game_seconds_remaining','down','ydstogo','yardline_100','score_differential']] = \
        df[['qtr','shotgun','game_seconds_remaining','down','ydstogo','yardline_100','score_differential']] \
        .apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    df['play_type_encoded'] = df['play_type'].apply(lambda x: 1 if x == "pass" else 0)
    df = df[df['posteam'] == "KC"]
    df_2024 = df[df['season'] == 2024]
    return pd.concat([df, df_2024, df_2024, df_2024], ignore_index=True)

df = load_data()

def train_or_load_models(df):
    def train(df, label, path):
        X = df[['qtr','game_seconds_remaining','down','ydstogo','yardline_100','score_differential']]
        y = df[label]
        model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        model.fit(X, y)
        model.save_model(path)
        return model

    # Train play type models
    def train_play_models(df):
        models = {}
        for sg in [0, 1]:
            subset = df[df['shotgun'] == sg]
            path = f"model_shotgun_{sg}.json"
            if os.path.exists(path):
                model = xgb.XGBClassifier()
                model.load_model(path)
            else:
                model = train(subset, 'play_type_encoded', path)
            models[sg] = model
        return models

    # Train shotgun prediction model
    if os.path.exists("model_shotgun_predict.json"):
        shotgun_model = xgb.XGBClassifier()
        shotgun_model.load_model("model_shotgun_predict.json")
    else:
        shotgun_model = train(df, 'shotgun', "model_shotgun_predict.json")

    return {
        "play_type_models": train_play_models(df),
        "shotgun_predict": shotgun_model
    }

models = train_or_load_models(df)

# Weighted player dictionaries
RUSHERS = {
    "Kareem Hunt": 200,
    "Isiah Pacheco": 83,
    "Patrick Mahomes": 58,
    "Carson Steele": 56,
    "Samaje Perine": 20,
    "Xavier Worthy (WR)": 20,
    "Mecole Hardman (WR)": 5
}

RECEIVERS = {
    "Travis Kelce": 97,
    "Xavier Worthy": 59,
    "Noah Gray": 40,
    "DeAndre Hopkins": 41,
    "Samaje Perine": 28,
    "Justin Watson": 22,
    "Rashee Rice": 24,
    "JuJu Smith-Schuster": 18,
    "Hollywood Brown": 9,
    "Mecole Hardman": 12,
    "Isiah Pacheco": 12
}

def deterministic_seed(*args):
    key = "-".join(str(x) for x in args)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (10**8)

def weighted_choice(choices, seed):
    np.random.seed(seed)
    keys, weights = zip(*choices.items())
    return np.random.choice(keys, p=np.array(weights) / sum(weights))

def predict_play_details(play_type, seed):
    np.random.seed(seed)
    if play_type == "run":
        direction = np.random.choice(["Left", "Middle", "Right"], p=[0.35, 0.4, 0.25])
        return f"{direction} run", weighted_choice(RUSHERS, seed + 1)
    else:
        depth = np.random.choice(["Short", "Deep"], p=[0.75, 0.25])
        side = np.random.choice(["Left", "Middle", "Right"], p=[0.3, 0.4, 0.3])
        return f"{depth} {side} pass", weighted_choice(RECEIVERS, seed + 2)

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

if go:
    game_time = (minutes * 60) + seconds
    X_input = np.array([[qtr, game_time, down, ydstogo, yardline, score_diff]])

    play_models = models['play_type_models']
    shotgun_predictor = models['shotgun_predict']

    prob_shotgun = shotgun_predictor.predict_proba(X_input)[0][1] * 100
    st.subheader("🔫 SHOTGUN FORMATION PREDICTION")
    st.write(f"🔄 Likelihood of using shotgun: **{prob_shotgun:.2f}%**")

    results = {}
    for sg in [0, 1]:
        model = play_models[sg]
        pass_prob = model.predict_proba(X_input)[0][1] * 100
        run_prob = 100 - pass_prob
        play_type = "pass" if pass_prob > run_prob else "run"
        seed = deterministic_seed(qtr, game_time, down, ydstogo, yardline, score_diff, sg)
        play_desc, player = predict_play_details(play_type, seed)

        results[sg] = {
            "pass": pass_prob,
            "run": run_prob,
            "desc": play_desc,
            "player": player
        }

    st.subheader("📈 PREDICTED PLAY DETAILS")
    st.markdown(f"**WITH SHOTGUN**: {results[1]['pass']:.2f}% PASS / {results[1]['run']:.2f}% RUN")
    st.write(f"🧠 Predicted play: {results[1]['desc']} to **{results[1]['player']}**")

    st.markdown("---")

    st.markdown(f"**WITHOUT SHOTGUN**: {results[0]['pass']:.2f}% PASS / {results[0]['run']:.2f}% RUN")
    st.write(f"🧠 Predicted play: {results[0]['desc']} to **{results[0]['player']}**")
