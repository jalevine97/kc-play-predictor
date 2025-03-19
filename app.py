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
        [data-testid="stSidebar"] { background-color: white !important; color: #00529B !important; }
        div[data-baseweb="select"] { border: 1px solid #00529B !important; border-radius: 5px !important; }
        .stSlider > div > div { color: #00529B !important; }
        html, body, [class*="css"] { font-family: Arial, sans-serif; }
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

        # Encode play_type
        df['play_type_encoded'] = df['play_type'].apply(lambda x: 1 if x == "pass" else 0)

        # Filter for KC offensive plays ONLY
        df = df[df['posteam'] == "KC"]

        # Weight 2024 plays more heavily
        df_2024 = df[df['season'] == 2024]
        df = pd.concat([df, df_2024, df_2024, df_2024], ignore_index=True)

        # Debugging - Check play count by opponent
        opponent_counts = df['defteam'].value_counts().to_dict()
        st.write("🔍 KC Play Count by Opponent:", opponent_counts)

        st.write(f"✅ Successfully loaded {len(df)} plays for KC from 2020-2024.")
        return df

    except Exception as e:
        st.error(f"🚨 Failed to load data: {e}")
        return None

# ✅ Load Data (Cached)
df = load_data()

if df is not None:
    # ✅ Sidebar Layout
    with st.sidebar:
        st.image("Eaglelogo2color.jpg", width=250)
        st.title("📊 Play Predictor - KC Chiefs")

        # 🏆 Select Week & Opponent
        week = st.selectbox("Select Game Week", list(KC_2024_SCHEDULE.keys()), index=7)
        opponent = KC_2024_SCHEDULE[week]

        if opponent == "BYE":
            st.warning("🚨 Kansas City has a BYE in Week 6. Select another week.")
        else:
            st.markdown(f"### 🏈 Opponent: **{opponent}**")

            # 🏈 Game Situation Inputs
            qtr = st.selectbox("Select Quarter", [1, 2, 3, 4], index=2)
            minutes = st.selectbox("Minutes Remaining", list(range(15, -1, -1)), index=1)
            seconds = st.slider("Seconds Remaining", min_value=0, max_value=59, value=14)
            down = st.selectbox("Down", [1, 2, 3, 4], index=0)
            ydstogo = st.slider("Yards to Go", min_value=1, max_value=30, value=10)
            yardline = st.slider("Field Position (0-50 KC Side, 50-100 Opponent Side)", 1, 99, 20)
            score_differential = st.slider("Score Differential (KC - Opponent)", -30, 30, 4)

    # ✅ Button to Trigger Prediction
    if st.sidebar.button("🔍 Get Prediction"):
        game_time = (minutes * 60) + seconds

        filtered_df = df[
            (df['qtr'] == qtr) &
            (df['down'] == down) &
            (df['game_seconds_remaining'].between(game_time - 1200, game_time + 1200)) &
            (df['ydstogo'].between(ydstogo - 10, ydstogo + 10)) &
            (df['yardline_100'].between(yardline - 10, yardline + 10)) &
            (df['score_differential'].between(score_differential - 10, score_differential + 10))
        ]

        st.write(f"✅ Final KC Play Count: {len(filtered_df)}")

        if len(filtered_df) < 10:
            st.error("🚨 Not enough KC plays found! Try adjusting filters.")
            st.stop()

        # ✅ Train XGBoost Model
        def train_xgb_model(df, shotgun):
            train_df = df[df['shotgun'] == shotgun]
            X = train_df[['qtr', 'game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'score_differential']]
            y = train_df['play_type_encoded']

            if len(y.unique()) < 2:
                return None

            model = xgb.XGBClassifier(eval_metric="logloss")
            model.fit(X, y)
            return model

        model_shotgun = train_xgb_model(filtered_df, shotgun=1)
        model_no_shotgun = train_xgb_model(filtered_df, shotgun=0)

        if model_shotgun is None or model_no_shotgun is None:
            st.error("🚨 Model training failed! Try different filters.")
            st.stop()

        # ✅ Debugging Logs
        st.write("📊 Model Training Completed!")
        st.write("🔎 Model Shotgun Exists:", model_shotgun is not None)
        st.write("🔎 Model No-Shotgun Exists:", model_no_shotgun is not None)

        # ✅ Predictions
        input_features = np.array([[qtr, game_time, down, ydstogo, yardline, score_differential]])
        st.write("📊 Model Input Features:", input_features)

        prediction_shotgun = model_shotgun.predict_proba(input_features)[0][1] * 100
        prediction_no_shotgun = model_no_shotgun.predict_proba(input_features)[0][1] * 100

        run_shotgun = 100 - prediction_shotgun
        run_no_shotgun = 100 - prediction_no_shotgun

        # ✅ Display Predictions
        st.subheader("🔮 Prediction Results:")
        st.write(f"📌 **With Shotgun:** {prediction_shotgun:.2f}% Pass, {run_shotgun:.2f}% Run")
        st.write(f"📌 **Without Shotgun:** {prediction_no_shotgun:.2f}% Pass, {run_no_shotgun:.2f}% Run")
