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

# ✅ Custom CSS for Theming
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: white !important; color: #01284a !important; }
        div[data-baseweb="select"] { border: 1px solid #01284a !important; border-radius: 5px !important; }
        .stSlider > div > div { color: #01284a !important; }
        html, body, [class*="css"] { font-family: 'proxima-sans', sans-serif; }
    </style>
""", unsafe_allow_html=True)


# ✅ Load Data Function (Cached)
@st.cache_data
def load_data():
    """Fetches NFL play-by-play data for 2020-2024 using sportsdataverse."""
    st.write("📡 Fetching latest data from sportsdataverse...")
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
        for col in ['qtr', 'shotgun', 'game_seconds_remaining', 'ydstogo', 'yardline_100', 'score_differential']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Encode play_type (Pass = 1, Run = 0)
        df['play_type_encoded'] = df['play_type'].apply(lambda x: 1 if x == "pass" else 0)

        # Filter for KC offensive plays ONLY
        df = df[df['posteam'] == "KC"]

        # Weight 2024 plays more heavily
        df_2024 = df[df['season'] == 2024]
        df = pd.concat([df, df_2024, df_2024, df_2024], ignore_index=True)

        st.write(f"✅ Successfully loaded {len(df)} plays for KC from 2020-2024.")
        return df

    except Exception as e:
        st.error(f"🚨 Failed to load data: {e}")
        return None

# ✅ Load Data (Cached)
df = load_data()


# ✅ Caching XGBoost Model Training – TRAIN ONCE PER SESSION
@st.cache_resource
def train_xgb_model(train_df):
    """Train an XGBoost model and cache it."""
    if len(train_df) < 10:
        return None

    X = train_df[['qtr', 'game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'score_differential']]
    y = train_df['play_type_encoded']

    if len(y.unique()) < 2:
        return None

    model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    model.fit(X, y)
    return model


if df is not None:
    # ✅ Sidebar Layout
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

        # **Field Position Adjustments**
        field_tolerance = 5 if yardline >= 80 else 15

        filtered_df = df[
            (df['qtr'] == qtr) &
            (df['down'] == down) &
            (df['game_seconds_remaining'].between(game_time - 600, game_time + 600)) &
            (df['ydstogo'].between(ydstogo - 10, ydstogo + 10)) &
            (df['yardline_100'].between(yardline - field_tolerance, yardline + field_tolerance)) &
            (df['score_differential'].between(score_differential - 10, score_differential + 10))
        ]

        st.write(f"✅ FINAL KC PLAY COUNT: {len(filtered_df)}")

        if len(filtered_df) < 10:
            st.error("🚨 NOT ENOUGH KC PLAYS FOUND! TRY ADJUSTING FILTERS.")
            st.stop()

        # ✅ Train Models ONCE for Shotgun & No-Shotgun
        model_shotgun = train_xgb_model(filtered_df[filtered_df['shotgun'] == 1])
        model_no_shotgun = train_xgb_model(filtered_df[filtered_df['shotgun'] == 0])

        if model_shotgun is None or model_no_shotgun is None:
            st.error("🚨 MODEL TRAINING FAILED! TRY DIFFERENT FILTERS.")
            st.stop()

        # ✅ Predictions
        input_features = np.array([[qtr, game_time, down, ydstogo, yardline, score_differential]])
        prediction_shotgun = model_shotgun.predict_proba(input_features)[0][1] * 100
        prediction_no_shotgun = model_no_shotgun.predict_proba(input_features)[0][1] * 100

        run_shotgun = 100 - prediction_shotgun
        run_no_shotgun = 100 - prediction_no_shotgun

        # ✅ Display Predictions
        st.subheader("🔮 PREDICTION RESULTS:")
        st.write(f"📌 **WITH SHOTGUN:** {prediction_shotgun:.2f}% PASS, {run_shotgun:.2f}% RUN")
        st.write(f"📌 **WITHOUT SHOTGUN:** {prediction_no_shotgun:.2f}% PASS, {run_no_shotgun:.2f}% RUN")
