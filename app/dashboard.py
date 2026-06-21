"""
dashboard.py — Streamlit demo for the Event-Driven Traffic Congestion Forecasting System.

Running on Colab:
    !pip install streamlit pyngrok -q
    from pyngrok import ngrok
    ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # free at ngrok.com
    !streamlit run app/dashboard.py &
    tunnel = ngrok.connect(8501)
    print(tunnel.public_url)

Running locally:
    streamlit run app/dashboard.py
"""
import os, sys, json, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.config import MODEL_PATH, FEEDBACK_LOG, RULES, CORRIDOR_CENTROIDS
from src.recommend import recommend, fallback_recommend

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Congestion Forecasting",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model ────────────────────────────────────────────────────────────────
CLOSURE_MODEL_PATH = MODEL_PATH.replace("severity_xgb", "closure_xgb")


@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None, None, None

    p_art = joblib.load(MODEL_PATH)
    priority_model = p_art["model"]
    p_threshold = p_art["threshold"]

    closure_model, c_threshold = None, None
    if os.path.exists(CLOSURE_MODEL_PATH):
        c_art = joblib.load(CLOSURE_MODEL_PATH)
        closure_model = c_art["model"]
        c_threshold = c_art["threshold"]

    builder_path = MODEL_PATH.replace("severity_xgb.pkl", "severity_xgb_builder.pkl")
    builder = joblib.load(builder_path) if os.path.exists(builder_path) else None

    metrics_path = os.path.join(os.path.dirname(MODEL_PATH), "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    return priority_model, p_threshold, closure_model, c_threshold, builder, metrics


@st.cache_data
def load_historical():
    parquet = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "data", "processed", "clean.parquet")
    if os.path.exists(parquet):
        return pd.read_parquet(parquet)
    return None


priority_model, p_threshold, closure_model, c_threshold, builder, metrics = load_artifact()
model, threshold = priority_model, p_threshold   # aliases for existing code below
hist_df = load_historical()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🚦 Traffic Forecasting")
tab = st.sidebar.radio("Navigate", ["Forecast & Recommend", "Historical Hotspots",
                                     "Simulated Live Feed", "Model Report"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Forecast & Recommend
# ═════════════════════════════════════════════════════════════════════════════
if tab == "Forecast & Recommend":
    st.title("Event Impact Forecast + Resource Recommendation")
    st.caption("Enter event details as they would be known at report time.")

    CORRIDOR_LIST = [
        "Non-corridor", "Mysore Road", "Bellary Road 1", "Bellary Road 2",
        "Tumkur Road", "Hosur Road", "Bannerghata Road", "ORR North 1",
        "ORR North 2", "ORR East 1", "ORR East 2", "ORR West 1",
        "Old Madras Road", "Magadi Road", "West of Chord Road",
        "Hennur Main Road", "CBD 1", "CBD 2", "Varthur Road",
        "Old Airport Road", "Airport New South Road", "IRR(Thanisandra road)"
    ]

    col1, col2 = st.columns(2)
    with col1:
        event_type   = st.selectbox("Event Type", ["unplanned", "planned"])
        event_cause  = st.selectbox("Event Cause", [
            "vehicle_breakdown", "accident", "construction", "congestion",
            "public_event", "procession", "protest", "vip_movement",
            "tree_fall", "water_logging", "pot_holes", "road_conditions", "others"
        ])
        requires_closure = st.checkbox("Requires Road Closure?")
        veh_type     = st.selectbox("Vehicle Type (if applicable)",
                                    ["none", "bmtc_bus", "heavy_vehicle", "lcv",
                                     "private_bus", "private_car", "ksrtc_bus",
                                     "truck", "taxi", "auto", "others"])
    with col2:
        # ── Corridor selection — drives lat/lon auto-fill ─────────────────────
        # Track previous corridor in session state to detect changes
        if "corridor_sel" not in st.session_state:
            st.session_state.corridor_sel = "Non-corridor"
        if "lat_val" not in st.session_state:
            st.session_state.lat_val = CORRIDOR_CENTROIDS["Non-corridor"][0]
        if "lon_val" not in st.session_state:
            st.session_state.lon_val = CORRIDOR_CENTROIDS["Non-corridor"][1]

        def on_corridor_change():
            new_corridor = st.session_state["corridor_widget"]
            if new_corridor != st.session_state.corridor_sel:
                centroid = CORRIDOR_CENTROIDS.get(new_corridor,
                           CORRIDOR_CENTROIDS["Non-corridor"])
                st.session_state.lat_val = centroid[0]
                st.session_state.lon_val = centroid[1]
                st.session_state.corridor_sel = new_corridor

        corridor = st.selectbox(
            "Corridor (lat/lon auto-fills to corridor centroid)",
            CORRIDOR_LIST,
            key="corridor_widget",
            on_change=on_corridor_change,
        )
        hour        = st.slider("Hour of Day (IST)", 0, 23, 8)
        day_of_week = st.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        dup_size    = st.number_input("Repeat-report cluster size", min_value=1, max_value=50, value=1)

    # Lat/lon: auto-populated from corridor centroid, still manually editable
    coord_col1, coord_col2 = st.columns(2)
    with coord_col1:
        lat = st.number_input(
            "Latitude (auto-filled from corridor, editable)",
            value=float(st.session_state.lat_val),
            format="%.6f",
            key="lat_input",
        )
    with coord_col2:
        lon = st.number_input(
            "Longitude (auto-filled from corridor, editable)",
            value=float(st.session_state.lon_val),
            format="%.6f",
            key="lon_input",
        )
    st.caption(f"📍 Using centroid for **{corridor}**: ({st.session_state.lat_val:.4f}, {st.session_state.lon_val:.4f})")

    dow_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
    is_peak = hour in {5,6,19,20,21,22}

    if st.button("Predict Severity & Get Recommendations", type="primary"):
        # ── Build single-row feature dict for rule engine directly if model unavailable
        proba = None
        label = None
        fallback_used = False

        if model is not None and builder is not None:
            try:
                # Build minimal feature row matching training schema
                row = pd.DataFrame([{
                    "event_type": event_type,
                    "event_cause": event_cause,
                    "requires_road_closure_bool": requires_closure,
                    "veh_type": veh_type,
                    "corridor": corridor,
                    "zone": "unknown",
                    "gba_identifier": "unknown",
                    "police_station": "unknown",
                    "junction": "none",
                    "has_junction": 0,
                    "dup_cluster_size": dup_size,
                    "lat": lat, "lon": lon,
                    "priority_high": 0,  # dummy
                    "start_datetime": pd.Timestamp.now(tz="Asia/Kolkata").replace(hour=hour),
                }])
                X, _ = builder.transform(row)
                proba_arr = model.predict_proba(X)[:, 1]
                proba = float(proba_arr[0])
                label = "High" if proba >= threshold else "Low"
            except Exception as e:
                st.warning(f"Model inference failed ({e}). Using fallback.")
                fallback_used = True
        else:
            fallback_used = True

        if fallback_used or (proba is not None and 0.4 <= proba <= 0.6):
            result = fallback_recommend(event_cause, requires_road_closure=requires_closure)
            label = result["severity"]
            proba = RULES["cause_high_rate"].get(event_cause, 0.5)
            result["fallback_used"] = True
        else:
            is_corridor_flag = 0 if corridor == "Non-corridor" else 1
            result = recommend({
                "severity": label,
                "probability": proba,
                "event_cause": event_cause,
                "requires_road_closure": requires_closure,
                "is_corridor": is_corridor_flag,
                "event_type": event_type,
                "hour": hour,
                "hour_bucket": "peak" if is_peak else "off-peak",
                "dup_cluster_size": dup_size,
                "corridor_name": corridor,
            })
            result["fallback_used"] = False

        # ── Model 2: Road closure prediction ──────────────────────────────────
        closure_pred, closure_proba = None, None
        if closure_model is not None and builder is not None and not fallback_used:
            try:
                c_labels, c_probas = closure_model.predict_proba(X)[:, 1], None
                c_probas = closure_model.predict_proba(X)[:, 1]
                closure_proba = float(c_probas[0])
                closure_pred = closure_proba >= c_threshold
            except Exception:
                pass

        # ── Display prediction ────────────────────────────────────────────────
        st.divider()
        st.subheader("Model Predictions")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Severity (Model 1)", label)
        c2.metric("Severity Confidence", f"{proba:.1%}" if proba is not None else "Fallback")
        if closure_pred is not None:
            c3.metric("Road Closure (Model 2)", "YES" if closure_pred else "No")
            c4.metric("Closure Confidence", f"{closure_proba:.1%}")
        else:
            c3.metric("Road Closure", "YES" if requires_closure else "No (input)")
            c4.metric("Threshold", f"{threshold:.2f}" if threshold else "N/A")

        if result.get("fallback_used"):
            st.info("ℹ️ Fallback rule used (model unavailable or low-confidence zone)")

        st.subheader("Resource Recommendation")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Manpower Required", result["manpower_count"])
        rc2.metric("Barricades", result["barricade_count"])
        rc3.metric("Diversion", "YES" if result["diversion_suggested"] else "No")

        st.markdown(f"**Barricade Placement:** {result['barricade_placement']}")
        st.markdown(f"**Diversion Note:** {result['diversion_note']}")

        with st.expander("Decision Rationale (fired rules)"):
            for rule in result["rationale"]:
                st.write(f"• {rule}")

        # ── Append to feedback log (same schema as notebook cell 7b) ─────────────
        log_row = {
            "event_id":               f"UI-{datetime.datetime.now().isoformat()}",
            "timestamp":              datetime.datetime.now().isoformat(),
            "event_cause":            event_cause,
            "predicted_severity":     label,
            "predicted_prob":         round(proba, 4) if proba else None,
            "recommended_manpower":   result["manpower_count"],
            "recommended_barricades": result["barricade_count"],
            "diversion_suggested":    result["diversion_suggested"],
            "actual_severity":        "",
            "actual_resolution_mins": "",
            "actual_manpower_used":   "",
            "operator_override":      "",
            "notes":                  "",
            "logged_at":              datetime.datetime.now().isoformat(),
        }
        log_df = pd.DataFrame([log_row])
        write_header = not os.path.exists(FEEDBACK_LOG)
        log_df.to_csv(FEEDBACK_LOG, mode="a", header=write_header, index=False)
        st.caption(f"Prediction logged to feedback_log.csv")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Historical Hotspots
# ═════════════════════════════════════════════════════════════════════════════
elif tab == "Historical Hotspots":
    st.title("Historical Incident Hotspots — Bengaluru")
    if hist_df is not None:
        st.metric("Total Incidents", len(hist_df))
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Events by Cause")
            cause_counts = hist_df["event_cause"].value_counts().head(10)
            fig, ax = plt.subplots()
            cause_counts.plot.barh(ax=ax)
            ax.set_xlabel("Count"); ax.set_title("Top 10 Event Causes")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("High vs Low Priority")
            priority_counts = hist_df["priority_high"].map({1:"High",0:"Low"}).value_counts()
            fig2, ax2 = plt.subplots()
            priority_counts.plot.pie(ax=ax2, autopct="%1.1f%%", startangle=90)
            ax2.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        st.subheader("Incident Map (Bengaluru)")
        map_df = hist_df[["lat","lon","priority_high"]].dropna()
        map_df = map_df.rename(columns={"lat":"latitude","lon":"longitude"})
        st.map(map_df.sample(min(500, len(map_df))))
    else:
        st.warning("Run the pipeline first to generate data/processed/clean.parquet")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Simulated Live Feed
# ═════════════════════════════════════════════════════════════════════════════
elif tab == "Simulated Live Feed":
    st.title("Simulated Live Event Feed")
    st.warning("⚠️ SIMULATED — This replays historical test-window events, not a real live feed.")

    if hist_df is not None and model is not None and builder is not None:
        from src.split import time_split
        _, _, test_df = time_split(hist_df)
        n_events = st.slider("Number of events to replay", 5, 50, 20)

        if st.button("Start Replay"):
            from src.realtime_sim import simulate_stream
            from src.model import predict as model_predict
            events_container = st.empty()
            events_data = []
            for ev in simulate_stream(test_df, builder, model, threshold, delay=0, n=n_events):
                events_data.append({
                    "Time": ev["timestamp"][:19],
                    "Cause": ev["event_cause"],
                    "Predicted": ev["predicted_severity"],
                    "Actual": ev["actual_priority"],
                    "Probability": f"{ev['probability']:.3f}",
                    "Manpower": ev["recommendation"]["manpower_count"],
                })
                events_container.dataframe(pd.DataFrame(events_data), use_container_width=True)
    else:
        st.warning("Train the model first (run the Colab notebook).")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: Model Report
# ═════════════════════════════════════════════════════════════════════════════
elif tab == "Model Report":
    st.title("Model Evaluation Report")
    models_dir = os.path.dirname(MODEL_PATH)

    if metrics:
        p = metrics.get("priority", {})
        cl = metrics.get("closure", {})
        pt = p.get("test", {}); pv = p.get("val", {})
        ct = cl.get("test", {}); cv = cl.get("val", {})

        st.subheader("Model 1 — Priority Severity (High vs Low)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test F1",      f"{pt.get('f1_pos', 0):.4f}")
        c2.metric("Test PR-AUC",  f"{pt.get('prauc', 0):.4f}")
        c3.metric("Val F1",       f"{pv.get('f1_pos', 0):.4f}")
        c4.metric("Threshold",    f"{p.get('threshold', 0.5):.3f}")

        if ct:
            st.subheader("Model 2 — Road Closure (True vs False)")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Test F1 (closure)", f"{ct.get('f1_pos', 0):.4f}")
            d2.metric("Test PR-AUC",       f"{ct.get('prauc', 0):.4f}")
            d3.metric("Val F1",            f"{cv.get('f1_pos', 0):.4f}")
            d4.metric("Threshold",         f"{cl.get('threshold', 0.5):.3f}")
    else:
        st.info("Train both models to see metrics (run the Colab notebook).")

    col1, col2 = st.columns(2)
    for fname, caption in [
        ("priority_feature_importance.png", "Priority — Feature Importances"),
        ("priority_confusion_matrix.png",   "Priority — Confusion Matrix (Test)"),
        ("closure_feature_importance.png",  "Closure — Feature Importances"),
        ("closure_confusion_matrix.png",    "Closure — Confusion Matrix (Test)"),
    ]:
        fpath = os.path.join(models_dir, fname)
        if os.path.exists(fpath):
            col1.image(fpath, caption=caption, use_column_width=True)
            col1, col2 = col2, col1  # alternate columns

    st.subheader("Feedback Log")
    if os.path.exists(FEEDBACK_LOG):
        flog = pd.read_csv(FEEDBACK_LOG)
        total = len(flog)
        filled = flog["actual_severity"].astype(str).str.strip().ne("").sum()
        st.dataframe(flog, use_container_width=True)
        st.caption(f"{total} predictions logged | {filled} with actual outcome filled")
    else:
        st.info("No predictions logged yet. Use the Forecast tab.")
