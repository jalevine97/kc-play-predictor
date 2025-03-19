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
        /* Sidebar Background and Text */
        [data-testid="stSidebar"] {
            background-color: white !important;
            color: #00529B !important;
        }
        
        /* Sidebar Titles */
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
            color: #00529B !important;
        }
        
        /* Sidebar Labels */
        .stSidebar label {
            color: #00529B !important;
            font-weight: bold;
        }
        
        /* Dropdown Borders */
        div[data-baseweb="select"] {
            border: 1px solid #00529B !important;
            border-radius: 5px !important;
        }

        /* Slider Track (Scroller) */
        .stSlider > div > div {
            color: #00529B !important;
        }

        /* Custom Font for Readability */
        html, body, [class*="css"] {
            font-family: Arial, sans-serif;
        }
    </style>
""", unsafe_allow_html=True)


# ✅ Load Data Function (Cached)
@st.cache_data
def load_data():
    """Fetches NFL play-by-play data for 2020-2024 using sportsdataverse."""
    st.write("📡 Fetching latest data from sportsdataverse...")
    try:
        raw_data = load_nfl_pbp(seasons=[2020, 2021, 2022, 2023, 2024])

        # Convert Polars DataFrame to Pandas
        if isinstance(raw_data, pl.DataFrame):
            df = raw_data.to_pandas()
        else:
            st.error("🚨 Unexpected data format received. Cannot proceed.")
            return None

        # Select relevant columns
        required_columns = ['season', 'week', 'qtr', 'game_seconds_remaining', 'down',
                            'ydstogo', 'yardline_100', 'score_differential', 'play_type', 
                            'shotgun', 'defteam', 'posteam']

        df = df[required_columns].dropna()  # ✅ Drop missing values

        # Ensure numeric conversion
        for col in ['qtr', 'shotgun', 'game_seconds_remaining', 'ydstogo', 'yardline_100', 'score_differential']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Encode play_type: Pass = 1, Run = 0
        df['play_type_encoded'] = df['play_type'].apply(lambda x: 1 if x == "pass" else 0)

        # Filter for KC offensive plays ONLY
        df = df[df['posteam'] == "KC"]

        # Weight 2024 plays more heavily
        df_2024 = df[df['season'] == 2024]
        df = pd.concat([df, df_2024, df_2024, df_2024], ignore_index=True)  # 4x weight for 2024

        # 🔎 Debugging - Check play count by opponent
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
    # ✅ Sidebar Layout with Updated Styles
    with st.sidebar:
        st.image("Eaglelogo2color.jpg", width=250)
        st.title("📊 Play Predictor - KC Chiefs")

        # 🏆 Select Week (Updated Schedule & Default: Week 8)
        week = st.selectbox("Select Game Week", list(KC_2024_SCHEDULE.keys()), index=7)
        opponent = KC_2024_SCHEDULE[week]

        if opponent == "BYE":
            st.warning("🚨 Kansas City has a BYE in Week 6. Select another week.")
        else:
            st.markdown(f"### 🏈 Opponent: **{opponent}**")

            # 🏈 Quarter (Default: 3rd)
            qtr = st.selectbox("Select Quarter", [1, 2, 3, 4], index=2)

            # ⏳ Game Time (Default: 14:14 left)
            minutes = st.selectbox("Minutes Remaining", list(range(15, -1, -1)), index=1)
            seconds = st.slider("Seconds Remaining", min_value=0, max_value=59, value=14)

            # 🔽 Down (Default: 1st)
            down = st.selectbox("Down", [1, 2, 3, 4], index=0)

            # 📏 Distance (Default: 10 yards)
            ydstogo = st.slider("Yards to Go", min_value=1, max_value=30, value=10)

            # 🏟️ Field Position (Default: KC 20 → yardline_100 = 20)
            yardline = st.slider(
                "Field Position (0-50 KC Side, 50-100 Opponent Side)", 
                min_value=1, max_value=99, value=20,
                help="0-50 is in KC's territory, 50-100 is in opponent's territory."
            )

            # 📉 Score Differential (Default: +4)
            score_differential = st.slider("Score Differential (KC - Opponent)", min_value=-30, max_value=30, value=4)

    # ✅ Button to Trigger Prediction
    if st.sidebar.button("🔍 Get Prediction"):
        st.subheader("🔮 Prediction Results:")
        st.write("✅ Data Found! Predictions will be added here.")
