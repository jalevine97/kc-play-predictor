# ✅ Button to Trigger Prediction
if st.sidebar.button("🔍 Get Prediction"):
    game_time = (minutes * 60) + seconds

    # Adjust field position tolerance dynamically
    field_position_tolerance = 10 if yardline <= 20 else 15

    filtered_df = df[
        (df['qtr'] == qtr) &
        (df['down'] == down) &
        (df['game_seconds_remaining'].between(game_time - 1200, game_time + 1200)) &
        (df['ydstogo'].between(ydstogo - 10, ydstogo + 10)) &
        (df['yardline_100'].between(yardline - field_position_tolerance, yardline + field_position_tolerance)) &
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
