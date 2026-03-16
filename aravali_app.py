import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import plotly.express as px
import geemap
from scipy.ndimage import zoom
from geopy.geocoders import Nominatim

# ---------------------------
# 1️⃣ INITIALIZATION & AUTH
# ---------------------------
try:
    service_account = st.secrets["GEE_SERVICE_ACCOUNT"]
    private_key = st.secrets["GEE_PRIVATE_KEY"]
    credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
    ee.Initialize(credentials)
except Exception as e:
    st.error(f"GEE Auth Error: {e}")

st.set_page_config(page_title="Aravali Forest Intelligence", layout="wide")
geolocator = Nominatim(user_agent="aravali_final_v15")

# ---------------------------
# 2️⃣ ADVANCED MODEL ARCHITECTURES
# ---------------------------
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
        outputs = layers.Dense(num_classes, activation='softmax', name="class_output")(x)
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
        x_reshaped = layers.Reshape((input_shape[0], -1))(x)
        scores = layers.TimeDistributed(layers.Dense(1))(x_reshaped)
        scores = layers.Flatten()(scores)
        weights = layers.Activation('softmax')(scores)
        weights = layers.Reshape((input_shape[0], 1))(weights)
        context_vector = layers.Multiply()([x_reshaped, weights])
        x_final = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
        x = layers.Dense(128, activation='relu')(x_final)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='linear')(x)
        return models.Model(inputs, output)

    c_model = build_improved_classification_model()
    r_model = build_regression_model_with_attention()
    c_model.load_weights("aravli_classifier_v2.weights.h5")
    r_model.load_weights("aravali_forest_model_v2.weights.h5")
    return c_model, r_model

# ---------------------------
# 3️⃣ UTILITIES & DYNAMIC DATA FETCHING
# ---------------------------
@st.cache_data(show_spinner=False)
def get_place_name(lat, lon):
    try:
        location = geolocator.reverse(f"{lat}, {lon}", timeout=3)
        return location.raw.get('address', {}).get('village', "Aravali Range") if location else "Aravali Range"
    except: return f"{lat}, {lon}"

def get_gee_sequence_and_trends(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(320).bounds()
    years = range(2020, 2027)
    sequence = []
    yearly_health = []

    for year in years:
        # Get Satellite Image
        img = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(region).filterDate(f"{year}-01-01", f"{year}-03-31")
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
               .median().clip(region))
        
        # Calculate NDVI for the graph: (B8 - B4) / (B8 + B4)
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        avg_ndvi = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10).get('NDVI').getInfo()
        yearly_health.append(avg_ndvi if avg_ndvi else 0.1)

        # Prepare Patch for AI (B4, B3, B2, B8)
        patch_img = img.select(['B4','B3','B2','B8'])
        patch = geemap.ee_to_numpy(patch_img, region=region, scale=10)
        if patch.shape[:2] != (64, 64):
            patch = zoom(patch, (64/patch.shape[0], 64/patch.shape[1], 1))
        sequence.append(patch)
        
    return np.array(sequence), yearly_health

# ---------------------------
# 4️⃣ CATEGORY DATA
# ---------------------------
CATEGORY_DATA = {
    "Healthy/Stable Forest": {"color": "#2ecc71", "consequences": ["Stable native biodiversity.", "Maximum groundwater recharge."], "steps": ["Protect existing canopy.", "Wildlife monitoring."], "impact": "✨ Maintains local cooling."},
    "Degraded/Encroached Forest": {"color": "#f1c40f", "consequences": ["Corridor fragmentation.", "Topsoil erosion."], "steps": ["Native Dhau plantation.", "Removal of invasive Prosopis."], "impact": "✨ Restores biodiversity corridors."},
    "Active Quarry/Mined Site": {"color": "#e74c3c", "consequences": ["Topographical destruction.", "Toxic silica dust."], "steps": ["Immediate closure.", "Step-terracing for stability."], "impact": "✨ Reduces dangerous dust pollution."},
    "Severe Land Degradation": {"color": "#d35400", "consequences": ["Total loss of biomass.", "Intense local heat."], "steps": ["Soil nutrient enrichment.", "Drip irrigation setup."], "impact": "✨ Prevents NCR dust storms."},
    "Stable Scrub Land": {"color": "#3498db", "consequences": ["Fire risk during May/June.", "Endemic fauna habitat."], "steps": ["Fire line management.", "Controlled grazing."], "impact": "✨ Protects grassland species."}
}

# ---------------------------
# 5️⃣ SIDEBAR & UI
# ---------------------------
st.sidebar.title("📍 Monitoring Site")
locations = {
    "Mangar Bani": [28.3500, 77.1731],
    "Pali Hills": [28.3120, 77.2050],
    "Sohna Ridge": [28.2200, 77.0620],
    "Nekpur, Faridabad": [28.3680, 77.2410],
    "Custom": None
}
loc_choice = st.sidebar.selectbox("Choose Site", list(locations.keys()))

if loc_choice == "Custom":
    lat = st.sidebar.number_input("Lat", value=28.3415, format="%.4f")
    lon = st.sidebar.number_input("Lon", value=77.2110, format="%.4f")
else:
    lat, lon = locations[loc_choice]

if st.sidebar.button("Run Deep Analysis"):
    with st.spinner("Analyzing Satellite Data with Attention AI..."):
        try:
            # FETCH ACTUAL DATA AND NDVI TRENDS
            raw_data, yearly_health = get_gee_sequence_and_trends(lat, lon)
            X_input = np.expand_dims(raw_data / (np.max(raw_data) + 1e-10), axis=0)

            c_model, r_model = load_all_models()
            class_probs = c_model.predict(X_input)
            loss_val = float(r_model.predict(X_input)[0][0])
            loss_val = -abs(loss_val)

            class_names = ["Mining/Degradation", "Scrub Land", "Forest"]
            initial_pred = class_names[np.argmax(class_probs)]

            if initial_pred == "Forest":
                final_status = "Degraded/Encroached Forest" if abs(loss_val) > 0.15 else "Healthy/Stable Forest"
            elif initial_pred == "Scrub Land":
                final_status = "Severe Land Degradation" if abs(loss_val) > 0.25 else "Stable Scrub Land"
            else:
                final_status = "Active Quarry/Mined Site"

            st.session_state.update({
                'analysis_done': True, 'final_status': final_status,
                'loss_val': loss_val, 'confidence': np.max(class_probs) * 100,
                'run_name': get_place_name(lat, lon), 'run_coords': [lat, lon],
                'yearly_health': yearly_health
            })
        except Exception as e:
            st.error(f"Analysis Failed: {e}")

# ---------------------------
# 6️⃣ RESULTS SECTION
# ---------------------------
if st.session_state.get('analysis_done'):
    st.header(f"Report: {st.session_state.run_name}")
    col1, col2 = st.columns(2)
    
    with col1:
        m = folium.Map(location=st.session_state.run_coords, zoom_start=14)
        folium.Marker(st.session_state.run_coords).add_to(m)
        st_folium(m, height=400, width=550, key="map_done")

    with col2:
        info = CATEGORY_DATA.get(st.session_state.final_status)
        st.markdown(f"<div style='background-color:{info['color']}; padding:20px; border-radius:10px; color:white; font-size:24px; text-align:center; font-weight:bold;'>{st.session_state.final_status.upper()}</div>", unsafe_allow_html=True)
        st.write("")
        st.metric("Measured Forest Loss Score", f"{st.session_state.loss_val:.4f}")
        st.write(f"AI Prediction Confidence: {st.session_state.confidence:.2f}%")
        st.subheader("🔴 Consequences")
        for c in info['consequences']: st.markdown(f"* {c}")

    st.divider()
    
    # DYNAMIC GRAPH: Using actual NDVI health values from GEE
    st.subheader("📊 7-Year Ecological Health Trend (Direct Satellite Measurement)")
    df_trend = pd.DataFrame({
        "Year": list(range(2020, 2027)), 
        "Health Index": st.session_state.yearly_health
    })
    
    # 
    fig = px.line(df_trend, x="Year", y="Health Index", markers=True, range_y=[0, 1])
    fig.update_traces(line_color=info['color'], line_width=4)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🛠️ Restoration Strategy")
    rest_col1, rest_col2 = st.columns(2)
    with rest_col1:
        st.markdown("**Recommended Steps:**")
        for step in info['steps']: st.markdown(f"✅ {step}")
    with rest_col2:
        st.markdown("**Impact of Restoration:**")
        st.markdown(f"{info['impact']}")

    csv_data = pd.DataFrame({
        "Year": range(2020, 2027),
        "Health_Index": st.session_state.yearly_health
    })
    st.download_button("📥 Export Health Data", csv_data.to_csv(index=False).encode('utf-8'), "aravali_trend.csv", "text/csv")
else:
    st.title("🛰️ Aravali Range Forest & Mining Monitor")
    st.info("Select a site from the sidebar and click 'Run Deep Analysis'.")