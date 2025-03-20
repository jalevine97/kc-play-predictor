import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import polars as pl
from sportsdataverse.nfl import load_nfl_pbp

# ✅ 2024 Opponent Schedule with Game Week Mapping (Updated)
KC_2024_SCHEDULE = {
    1: "BAL", 2: "CIN", 3: "ATL", 4: "LAC", 5: "NO",
    6: "BYE",  # Week 6 Bye
    7: "SF", 8: "LV", 9: "TB", 10: "DEN", 11: "BUF",
    12: "CAR", 13: "LV", 14: "LAC", 15: "CLE", 16: "HOU", 17: "PIT", 18: "DEN"
}

# ✅ Custom CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: white !important; color: #01284a !important; }
        div[data-baseweb="select"] { border: 1px solid #01284a !important; border-radius: 5px !important; }
        .stSlider > div > div { color: #01284a !important; }
        html, body, [class*="css"] { font-family: 'Proxima Nova', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# ✅ Load Data Function (Cached)
@st.cache_data
def load_data():
    """Fetches and processes KC play-by-play data for 2020-2024."""
    st.write("📡 Fetching latest KC play data...")
    try:
        raw_data = load_nfl_pbp(seasons=[2020, 2021, 2022, 2023, 2024])
        df = raw_data.to_pandas() if isinstance(raw_data, pl.DataFrame) else None
        if df is None: 
            st.error("🚨 Unexpected data format received. Cannot proceed.")
            return None

        # Select relevant columns
        cols = ['season', 'week', 'qtr', 'game_seconds_remaining', 'down',
                'ydstogo', 'yardline_100', 'score_differential', 'play_type', 
                'shotgun', 'defteam', 'posteam']
        df = df[cols].dropna()

        # Convert types
        numeric_cols = ['qtr', 'shotgun', 'game_seconds_remaining', 'ydstogo', 'yardline_100', 'score_differential']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

        # Encode play_type
        df['play_type_encoded'] = df['play_type'].apply(lambda x: 1 if x == "pass" else 0)

        # Filter for KC offensive plays ONLY
        df = df[df['posteam'] == "KC"]

        # Weight 2024 plays more heavily
        df_2024 = df[df['season'] == 2024]
        df = pd.concat([df, df_2024, df_2024, df_2024], ignore_index=True)

        st.write(f"✅ Successfully loaded {len(df)} KC plays from 2020-2024.")
        return df

    except Exception as e:
        st.error(f"🚨 Failed to load data: {e}")
        return None

# ✅ Load Data (Cached)
df = load_data()

# ✅ Train & Cache the Models **ONCE**
@st.cache_resource
def train_xgb_models(df):
    """Trains XGBoost models for shotgun and non-shotgun plays and caches them."""
    def train_xgb_model(train_df, shotgun):
        subset = train_df[train_df['shotgun'] == shotgun]
        X = subset[['qtr', 'game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'score_differential']]
        y = subset['play_type_encoded']

        if len(y.unique()) < 2:
            return None  # Not enough variation to train

        model = xgb.XGBClassifier(eval_metric="logloss")
        model.fit(X, y)
        return model

    return {
        "shotgun": train_xgb_model(df, shotgun=1),
        "no_shotgun": train_xgb_model(df, shotgun=0)
    }

# ✅ Train Once, Cache Globally
models = train_xgb_models(df)

# ✅ Sidebar Layout
if df is not None:
    with st.sidebar:
        st.image("Eaglelogo2color.jpg", width=250)
        st.title("📊 PLAY PREDICTOR - KC CHIEFS")

        # 🏆 Select Week & Opponent
        week = st.selectbox("SELECT GAME WEEK", list(KC_2024_SCHEDULE.keys()), index=7)
        opponent = KC_2024_SCHEDULE[week]

        if opponent == "BYE":
            st.warning("🚨 KANSAS CITY HAS A BYE IN WEEK 6. SELECT ANOTHER WEEK.")
        else:
            st.markdown(f"### 🏈 OPPONENT: **{opponent}**")

            # 🏈 Game Situation Inputs
            qtr = st.selectbox("SELECT QUARTER", [1, 2, 3, 4], index=2)
            minutes = st.selectbox("MINUTES REMAINING", list(range(15, -1, -1)), index=1)
            seconds = st.slider("SECONDS REMAINING", min_value=0, max_value=59, value=14)
            down = st.selectbox("DOWN", [1, 2, 3, 4], index=0)
            ydstogo = st.slider("YARDS TO GO", min_value=1, max_value=30, value=10)
            yardline = st.slider("FIELD POSITION (0-50 KC SIDE, 50-100 OPPONENT SIDE)", 1, 99, 20)
            score_differential = st.slider("SCORE DIFFERENTIAL (KC - OPPONENT)", -30, 30, 4)

    # ✅ Button to Trigger Prediction
    if st.sidebar.button("🔍 GET PREDICTION"):
        game_time = (minutes * 60) + seconds
        input_features = np.array([[qtr, game_time, down, ydstogo, yardline, score_differential]])

        model_shotgun = models["shotgun"]
        model_no_shotgun = models["no_shotgun"]

        if model_shotgun and model_no_shotgun:
            pass_shotgun = model_shotgun.predict_proba(input_features)[0][1] * 100
            run_shotgun = 100 - pass_shotgun

            pass_no_shotgun = model_no_shotgun.predict_proba(input_features)[0][1] * 100
            run_no_shotgun = 100 - pass_no_shotgun

            # ✅ Display Predictions
            st.subheader("🔮 PREDICTION RESULTS:")
            st.write(f"📌 **WITH SHOTGUN:** {pass_shotgun:.2f}% PASS, {run_shotgun:.2f}% RUN")
            st.write(f"📌 **WITHOUT SHOTGUN:** {pass_no_shotgun:.2f}% PASS, {run_no_shotgun:.2f}% RUN")
        else:
            st.error("🚨 MODEL TRAINING FAILED! TRY DIFFERENT FILTERS.")
