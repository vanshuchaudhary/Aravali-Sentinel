import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CORE LOGIC FUNCTIONS ---

def get_final_status(initial_pred, loss_val, f_thresh, s_thresh):
    abs_loss = abs(loss_val)
    status = initial_pred
    color = "white"

    if initial_pred == "Forest":
        if abs_loss > f_thresh:
            status, color = "Degraded/Encroached Forest", "orange"
        else:
            status, color = "Stable Forest", "green"
            
    elif initial_pred == "Scrub Land":
        if abs_loss > s_thresh:
            status, color = "Severe Land Degradation", "red"
        else:
            status, color = "Stable Scrub Land", "green"
            
    elif initial_pred == "Mining/Degradation":
        status, color = "Active Quarry/Mined Site", "red"
        
    return status, color

def calculate_projections(current_loss, target_year, model_type="Linear"):
    years_passed = 2026 - 2020
    annual_rate = abs(current_loss) / years_passed
    future_years = target_year - 2026
    
    if model_type == "Linear":
        projected = abs(current_loss) + (annual_rate * future_years)
    else:  # Accelerated (2% increase in rate per year)
        projected = abs(current_loss)
        temp_rate = annual_rate
        for _ in range(future_years):
            temp_rate *= 1.02 # 2% acceleration
            projected += temp_rate
            
    return min(projected, 1.0) # Cap at 100% loss

# --- 2. STREAMLIT UI SETUP ---

st.set_page_config(page_title="Aravali AI Watch", layout="wide")
st.title("🛡️ Aravali Eco-Intelligence Dashboard")
st.markdown("### Monitoring and Future Projections (2020-2050)")

# Sidebar Configuration
with st.sidebar:
    st.header("📍 Coordinate Input")
    lat = st.number_input("Latitude", value=28.812, format="%.4f")
    lon = st.number_input("Longitude", value=75.985, format="%.4f")
    
    st.header("⚙️ Model Thresholds")
    f_thresh = st.slider("Forest Degradation Trigger", 0.0, 0.5, 0.15)
    s_thresh = st.slider("Scrub Degradation Trigger", 0.0, 0.5, 0.25)
    
    st.header("🔮 Projection Settings")
    target_year = st.select_slider("Projection Year", options=[2030, 2040, 2050, 2060])
    model_type = st.radio("Projection Model", ["Linear", "Accelerated (2% Growth)"])
    
    run_btn = st.button("Generate Full Analysis", type="primary")

# --- 3. MAIN DASHBOARD EXECUTION ---

if run_btn:
    # MOCK DATA (Integrate your actual model calling code here)
    # Using your actual Dadam Hills result as a base
    initial_pred = "Forest" 
    loss_val = -0.2953
    confidence = 0.7545
    
    # Run Decision Logic
    final_status, status_color = get_final_status(initial_pred, loss_val, f_thresh, s_thresh)
    
    # Run Projections
    p_loss = calculate_projections(loss_val, target_year, model_type)

    # --- UI DISPLAY ---
    c1, c2, c3 = st.columns(3)
    c1.metric("2026 Status", final_status)
    c2.metric("Measured Loss Index", f"{loss_val:.4f}")
    c3.metric("Model Confidence", f"{confidence*100:.1f}%")

    st.divider()

    # Visual Comparison Subplot
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🛰️ Temporal Change Detection")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # (Replace with actual image loading logic)
        ax[0].set_title("2020 Baseline")
        ax[1].set_title(f"2026: {final_status}")
        st.pyplot(fig)

    with col_right:
        st.subheader(f"🔮 {target_year} Risk Assessment")
        st.write(f"Estimated Ecosystem Loss: **{p_loss*100:.1f}%**")
        st.progress(p_loss)
        
        if p_loss > 0.7:
            st.error(f"**CRITICAL RISK:** Total habitat collapse projected by {target_year}.")
        elif p_loss > 0.4:
            st.warning("**MODERATE RISK:** Significant fragmentation likely by target date.")
        else:
            st.success("**STABLE:** Ecosystem projected to retain majority of biomass.")

    st.info(f"**Insight:** Under the {model_type} model, this location loses land cover at a rate of {abs(loss_val/6)*100:.2f}% per year.")