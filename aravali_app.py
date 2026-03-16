import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import plotly.express as px
import ee
import geemap
from scipy.ndimage import zoom
from geopy.geocoders import Nominatim
import os
from dotenv import load_dotenv

# --- 1. INITIALIZATION ---
load_dotenv()
st.set_page_config(page_title="Aravali Forest Intelligence", layout="wide")

try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

geolocator = Nominatim(user_agent="aravali_monitor_final_v4")

def get_place_name(lat, lon):
    try:
        location = geolocator.reverse(f"{lat}, {lon}", timeout=3)
        if location:
            address = location.raw.get('address', {})
            village = address.get('village') or address.get('suburb') or address.get('hamlet') or "Aravali Range"
            city = address.get('city') or address.get('county') or ""
            return f"{village}, {city}"
        return f"Lat: {lat}, Lon: {lon}"
    except:
        return f"{lat}, {lon}"

# --- 2. GEE DATA FETCHING ---
def get_gee_sequence(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(320).bounds()
    years = range(2020, 2027)
    sequence = []

    for year in years:
        img = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(region)
               .filterDate(f"{year}-01-01", f"{year}-03-31")
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
               .median()
               .select(['B4','B3','B2','B8'])
               .clip(region))

        patch = geemap.ee_to_numpy(img, region=region, scale=10)
        if patch.shape[:2] != (64, 64):
            patch = zoom(patch, (64/patch.shape[0], 64/patch.shape[1], 1))
        sequence.append(patch)
    return np.array(sequence)

# --- 3. MODEL ARCHITECTURES ---
@st.cache_resource
def load_all_models():
    def build_improved_classification_model(input_shape=(7, 64, 64, 4), num_classes=3):
        inputs = layers.Input(shape=input_shape)
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x) 
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x) 
        att_weights = layers.Dense(1)(x)
        att_weights = layers.Softmax(axis=1)(att_weights)
        x_weighted = layers.Multiply()([x, att_weights])
        x_final = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x_weighted)
        x = layers.Dense(64, activation='relu', kernel_regularizer='l2')(x_final)
        x = layers.Dropout(0.6)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        return models.Model(inputs, outputs)

    def build_regression_model_with_attention(input_shape=(7, 64, 64, 4)):
        inputs = layers.Input(shape=input_shape)
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        time_steps = input_shape[0]
        x_reshaped = layers.Reshape((time_steps, -1))(x)
        scores = layers.TimeDistributed(layers.Dense(1))(x_reshaped)
        scores = layers.Flatten()(scores)
        weights = layers.Activation('softmax')(scores)
        weights = layers.Reshape((time_steps, 1))(weights)
        context_vector = layers.Multiply()([x_reshaped, weights])
        x_final = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
        x = layers.Dense(128, activation='relu')(x_final)
        output = layers.Dense(1, activation='linear')(x)
        return models.Model(inputs, output)

    c_model = build_improved_classification_model()
    c_model.load_weights("aravli_classifier_v2.weights.h5") 
    r_model = build_regression_model_with_attention()
    r_model.load_weights("aravali_forest_model_v2.weights.h5")
    return c_model, r_model

# --- 4. DATA ---
CATEGORY_DATA = {
    "Healthy/Stable Forest": {"color": "#2ecc71", "consequences": ["Stable native biodiversity.", "Maximum groundwater recharge."], "restoration_steps": ["Maintain strict protection."], "help_points": ["Prevents desertification."]},
    "Degraded/Encroached Forest": {"color": "#f1c40f", "consequences": ["Corridor fragmentation.", "Topsoil erosion."], "restoration_steps": ["Native Dhau afforestation."], "help_points": ["Restores local water table."]},
    "Active Quarry/Mined Site": {"color": "#e74c3c", "consequences": ["Topographical destruction.", "Toxic silica dust."], "restoration_steps": ["Mechanical pit reclamation."], "help_points": ["Restores safety and health."]},
    "Severe Land Degradation": {"color": "#d35400", "consequences": ["Total loss of biomass.", "Intense local heat."], "restoration_steps": ["Soil nutrient enrichment."], "help_points": ["Prevents NCR dust storms."]},
    "Stable Scrub Land": {"color": "#3498db", "consequences": ["Fire risk during May/June."], "restoration_steps": ["Fire line management."], "help_points": ["Protects endemic fauna."]}
}

# --- 5. SIDEBAR & LOGIC ---
st.sidebar.title("📍 Monitoring Site")
locations = {
    "Mangar Bani": [28.3500, 77.1731],
    "Pali Hills": [28.3120, 77.2050],
    "Sohna Ridge": [28.2200, 77.0620],
    "Custom": None
}
loc_choice = st.sidebar.selectbox("Choose Site", list(locations.keys()))

if loc_choice == "Custom":
    lat = st.sidebar.number_input("Lat", value=28.3415, format="%.4f")
    lon = st.sidebar.number_input("Lon", value=77.2110, format="%.4f")
    coords = [lat, lon]
else:
    coords = locations[loc_choice]
    lat, lon = coords

if st.sidebar.button("Run Deep Analysis"):
    with st.spinner("Analyzing 7-Year Satellite Sequence..."):
        raw_data = get_gee_sequence(lat, lon)
        X_input = raw_data / (np.max(raw_data) + 1e-10)
        X_input = np.expand_dims(X_input, axis=0) 

        c_model, r_model = load_all_models()
        class_probs = c_model.predict(X_input)
        loss_val = float(r_model.predict(X_input)[0][0])
        
        class_names = ["Mining/Degradation", "Scrub Land", "Forest"]
        pred_idx = np.argmax(class_probs)
        initial_pred = class_names[pred_idx]
        abs_loss = abs(loss_val)

        if initial_pred == "Forest":
            final_status = "Degraded/Encroached Forest" if abs_loss > 0.15 else "Healthy/Stable Forest"
        elif initial_pred == "Scrub Land":
            final_status = "Severe Land Degradation" if abs_loss > 0.25 else "Stable Scrub Land"
        elif initial_pred == "Mining/Degradation":
            final_status = "Active Quarry/Mined Site"
        else:
            final_status = initial_pred

        st.session_state.analysis_done = True
        st.session_state.final_status = final_status
        st.session_state.loss_val = loss_val
        st.session_state.confidence = np.max(class_probs) * 100
        st.session_state.run_name = get_place_name(lat, lon)
        st.session_state.run_coords = coords

# --- 6. DISPLAY RESULTS ---
if st.session_state.get('analysis_done'):
    st.header(f"Report: {st.session_state.run_name}")
    
    col1, col2 = st.columns(2)
    with col1:
        m = folium.Map(location=st.session_state.run_coords, zoom_start=14)
        folium.Marker(st.session_state.run_coords, popup=st.session_state.run_name).add_to(m)
        folium.Circle(st.session_state.run_coords, radius=320, color='red', fill=True, opacity=0.2).add_to(m)
        st_folium(m, height=400, width=550, key="map_done")

    with col2:
        info = CATEGORY_DATA.get(st.session_state.final_status, CATEGORY_DATA["Stable Scrub Land"])
        st.markdown(f"<div style='background-color:{info['color']}; padding:20px; border-radius:10px; color:white; font-size:24px; text-align:center;'><b>{st.session_state.final_status.upper()}</b></div>", unsafe_allow_html=True)
        
        st.write("---")
        st.metric("Measured Forest Loss Score", f"{st.session_state.loss_val:.4f}")
        st.write(f"**AI Prediction Confidence:** {st.session_state.confidence:.2f}%")
        
        st.subheader("🔴 Consequences")
        for c in info['consequences']: st.write(f"- {c}")

    st.divider()
    
    # --- DYNAMIC CHART SECTION ---
    st.subheader("📈 7-Year Ecological Health Trend")
    years = list(range(2020, 2027))
    total_change = st.session_state.loss_val
    health_trend = []
    base_start = 0.85 
    for i in range(7):
        current_val = base_start + (total_change * (i / 6))
        health_trend.append(max(0.05, min(1.0, current_val))) 

    df_trend = pd.DataFrame({"Year": years, "Health Index": health_trend})
    fig = px.line(df_trend, x="Year", y="Health Index", markers=True, range_y=[0, 1])
    fig.update_traces(line_color=info['color'], line_width=4, marker=dict(size=10))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🛠️ Restoration Strategy")
    r1, r2 = st.columns(2)
    with r1:
        st.write("**Recommended Steps:**")
        for s in info['restoration_steps']: st.write(f"✅ {s}")
    with r2:
        st.write("**Impact of Restoration:**")
        for h in info['help_points']: st.write(f"✨ {h}")

    # --- EXPORT BUTTON ---
    st.divider()
    report_df = pd.DataFrame({
        "Parameter": ["Location", "Latitude", "Longitude", "Status", "Loss Score", "AI Confidence"],
        "Value": [
            st.session_state.run_name, 
            st.session_state.run_coords[0], 
            st.session_state.run_coords[1], 
            st.session_state.final_status, 
            st.session_state.loss_val, 
            f"{st.session_state.confidence:.2f}%"
        ]
    })
    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Analysis Report (CSV)",
        data=csv,
        file_name=f"aravali_report_{st.session_state.run_name.replace(' ', '_')}.csv",
        mime='text/csv',
    )

else:
    st.title("🛰️ Aravali Range Forest & Mining Monitor")
    st.info("Pick a location and click 'Run Deep Analysis' to generate the report.")