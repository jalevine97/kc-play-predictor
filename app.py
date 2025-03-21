import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import polars as pl
from sportsdataverse.nfl import load_nfl_pbp
import os

# ✅ 2024 Opponent Schedule
KC_2024_SCHEDULE = {
    1: "BAL", 2: "CIN", 3: "ATL", 4: "LAC", 5: "NO",
    6: "BYE", 7: "SF", 8: "LV", 9: "TB", 10: "DEN", 11: "BUF",
    12: "CAR", 13: "LV", 14: "LAC", 15: "CLE", 16: "HOU", 17: "PIT", 18: "DEN"
}

# ✅ Adaptive CSS for Dark and Light Modes
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Proxima Nova', sans-serif;
        }

        /* Sidebar styles */
        [data-testid="stSidebar"] {
            background-color: white;
        }

        /* Text styles */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
        .stMarkdown h5, .stMarkdown h6, .stMarkdown p, .stText, .stTitle, .stSubheader {
            color: var(--text-color);
        }

        /* Dropdown, slider, and widget borders */
        div[data-baseweb="select"], .stSlider > div > div {
            border: 1px solid #01284a !important;
            border-radius: 5px !important;
        }

        /* Set theme-aware text color */
        @media (prefers-color-scheme: dark) {
            html, body, [class*="css"] {
                color: #e0e0e0;
                background-color: #0e1117;
            }

            [data-testid="stSidebar"] {
                background-color: #0e1117 !important;
            }
        }

        @media (prefers-color-scheme: light) {
            html, body, [class*="css"] {
                color: #01284a;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Load Data Function
@st.cache_data
def load_data():
    st.write("📡 Fetching latest KC play data...")
    try:
        raw_data = load_nfl_pbp(seasons=[2020, 2021, 2022, 2023, 2024])
        df = raw_data.to_pandas() if isinstance(raw_data, pl.DataFrame) else None
        if df is None:
            st.error("🚨 Unexpected data format received.")
            return None

        cols = ['season', 'week', 'qtr', 'game_seconds_remaining', 'down',
                'ydstogo', 'yardline_100', 'score_differential', 'play_type',
                'shotgun', 'defteam', 'posteam']
        df = df[cols].dropna()

        numeric_cols = ['qtr', 'shotgun', 'game_seconds_remaining', 'ydstogo', 'yardline_100', 'score_differential']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

        df['play_type_encoded'] = df['play_type'].apply(lambda x: 1 if x == "pass" else 0)
        df = df[df['posteam'] == "KC"]

        df_2024 = df[df['season'] == 2024]
        df = pd.concat([df, df_2024, df_2024, df_2024], ignore_index=True)

        st.write(f"✅ Successfully loaded {len(df)} KC plays from 2020-2024.")
        return df

    except Exception as e:
        st.error(f"🚨 Failed to load data: {e}")
        return None

df = load_data()

# ✅ Check if Model Files Exist
model_shotgun_path = "model_shotgun.json"
model_no_shotgun_path = "model_no_shotgun.json"

# ✅ Train and Save Models if Not Found
@st.cache_resource
def train_xgb_models(train_df):
    def train_model(df, shotgun, filename):
        subset = df[df['shotgun'] == shotgun]
        X = subset[['qtr', 'game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'score_differential']]
        y = subset['play_type_encoded']

        st.write(f"🔍 Training {'shotgun' if shotgun else 'no-shotgun'} model... ({len(subset)} plays)")
        st.write(f"📊 Label breakdown: {y.value_counts().to_dict()}")

        if len(y.unique()) < 2:
            st.warning(f"⚠️ Not enough variation in {'shotgun' if shotgun else 'no-shotgun'} plays. Skipping model.")
            return None

        model = xgb.XGBClassifier(eval_metric="logloss")
        model.fit(X, y)
        model.save_model(filename)
        return model

    # Paths to model files
    model_shotgun_path = "model_shotgun.json"
    model_no_shotgun_path = "model_no_shotgun.json"

    models = {}

    # Try loading pre-trained models
    if os.path.exists(model_shotgun_path):
        try:
            model = xgb.XGBClassifier()
            model.load_model(model_shotgun_path)
            models["shotgun"] = model
            st.write("✅ Loaded saved shotgun model.")
        except Exception as e:
            st.warning(f"⚠️ Could not load shotgun model: {e}")
            models["shotgun"] = None
    else:
        models["shotgun"] = train_model(train_df, shotgun=1, filename=model_shotgun_path)

    if os.path.exists(model_no_shotgun_path):
        try:
            model = xgb.XGBClassifier()
            model.load_model(model_no_shotgun_path)
            models["no_shotgun"] = model
            st.write("✅ Loaded saved no-shotgun model.")
        except Exception as e:
            st.warning(f"⚠️ Could not load no-shotgun model: {e}")
            models["no_shotgun"] = None
    else:
        models["no_shotgun"] = train_model(train_df, shotgun=0, filename=model_no_shotgun_path)

    return models

# ✅ Load models once
models = train_xgb_models(df)

# ✅ Sidebar Layout
if df is not None:
    with st.sidebar:
        st.image("Eaglelogo2color.jpg", width=250)
        st.title("📊 PLAY PREDICTOR - KC CHIEFS")
        week = st.selectbox("SELECT GAME WEEK", list(KC_2024_SCHEDULE.keys()), index=7)
        opponent = KC_2024_SCHEDULE[week]

        if opponent == "BYE":
            st.warning("🚨 BYE WEEK. SELECT ANOTHER.")
        else:
            st.markdown(f"### 🏈 OPPONENT: **{opponent}**")
            qtr = st.selectbox("SELECT QUARTER", [1, 2, 3, 4], index=2)
            minutes = st.selectbox("MINUTES REMAINING", list(range(15, -1, -1)), index=1)
            seconds = st.slider("SECONDS REMAINING", 0, 59, 14)
            down = st.selectbox("DOWN", [1, 2, 3, 4], index=0)
            ydstogo = st.slider("YARDS TO GO", 1, 30, 10)
            yardline = st.slider("FIELD POSITION", 1, 99, 20)
            score_differential = st.slider("SCORE DIFF (KC - OPP)", -30, 30, 4)
            submit = st.button("🔍 GET PREDICTION")

    if submit:
        game_time = (minutes * 60) + seconds
        input_data = np.array([[qtr, game_time, down, ydstogo, yardline, score_differential]])

        model_shotgun = models["shotgun"]
        model_no_shotgun = models["no_shotgun"]

        if model_shotgun and model_no_shotgun:
            pass_shotgun = model_shotgun.predict_proba(input_data)[0][1] * 100
            run_shotgun = 100 - pass_shotgun

            pass_no_shotgun = model_no_shotgun.predict_proba(input_data)[0][1] * 100
            run_no_shotgun = 100 - pass_no_shotgun

            st.subheader("🔮 PREDICTION RESULTS:")
            st.write(f"📌 **WITH SHOTGUN:** {pass_shotgun:.2f}% PASS, {run_shotgun:.2f}% RUN")
            st.write(f"📌 **WITHOUT SHOTGUN:** {pass_no_shotgun:.2f}% PASS, {run_no_shotgun:.2f}% RUN")
        else:
            st.error("🚨 MODEL TRAINING FAILED.")
