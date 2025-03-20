import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import polars as pl
from sportsdataverse.nfl import load_nfl_pbp

# ✅ Opponent Defensive Data - (Example Placeholder Data)
OPPONENT_DEFENSE_STATS = {
    "BAL": {"epa_per_play": -0.12, "run_def_rank": 2, "blitz_rate": 35},
    "CIN": {"epa_per_play": -0.08, "run_def_rank": 12, "blitz_rate": 20},
    "ATL": {"epa_per_play": 0.05, "run_def_rank": 28, "blitz_rate": 18},
    "LAC": {"epa_per_play": -0.02, "run_def_rank": 20, "blitz_rate": 30},
    "NO": {"epa_per_play": -0.10, "run_def_rank": 5, "blitz_rate": 33},
    # More teams can be added...
}

# ✅ Function to Fetch Opponent Data
def get_opponent_defense_stats(opponent):
    return OPPONENT_DEFENSE_STATS.get(opponent, {"epa_per_play": 0, "run_def_rank": 16, "blitz_rate": 25})

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
            opponent_stats = get_opponent_defense_stats(opponent)
            st.write(f"🛡️ **Defense EPA:** {opponent_stats['epa_per_play']}")
            st.write(f"🚀 **Blitz Rate:** {opponent_stats['blitz_rate']}%")
            st.write(f"🏈 **Run Defense Rank:** {opponent_stats['run_def_rank']}")

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
