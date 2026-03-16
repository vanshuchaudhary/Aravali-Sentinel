# 🌲 Aravali Forest Intelligence (AFI) 🛰️
### **Real-Time AI-Powered Satellite Surveillance for the Aravalli Range**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://vanshuchaudhary-aravali-sentinel-aravali-app-fmcsqj.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The Aravalli Range is one of the world's oldest mountain ranges, acting as a critical green barrier against the Thar Desert. **AFI** uses Deep Learning (CNN-BiLSTM) and Google Earth Engine to provide real-time ecological health monitoring and risk assessment.

---

## 🌟 Key Features
* **🤖 Precision AI Analysis:** Custom-trained neural networks provide site-specific classification with high confidence scores.
* **📈 7-Year Temporal Tracking:** Direct satellite measurement of forest health trends from 2020 to 2026.
* **🗺️ Risk Hotspot Mapping:** Identification of degraded and encroached forest zones using multi-spectral imagery.
* **🛠️ Restoration Intelligence:** Automated recommendations for native plantation and biodiversity restoration.

---

## 📸 Project Showcase

### **1. AI-Powered Monitoring Dashboard**
The engine identifies forest status at specific coordinates. In this example, the AI detects a degraded zone with high precision, highlighting topsoil erosion and corridor fragmentation risks.
![AI Analysis](screenshots/Screenshot%20(164).jpg)

### **2. 7-Year Ecological Health Trends**
Tracking the "heartbeat" of the forest. This graph uses direct satellite measurements to visualize how vegetation health has fluctuated over nearly a decade.
![Temporal Trends](screenshots/Screenshot%20(165).png)

### **3. Smart Restoration Strategies**
The system doesn't just find problems; it suggests solutions. From recommending **Native Dhau** plantation to identifying invasive species, it provides an actionable roadmap for conservation.
![Restoration Strategy](screenshots/Screenshot%20(166).jpg)

---

## 🧠 Tech Stack
* **Frontend:** Streamlit
* **Satellite Engine:** Google Earth Engine (Sentinel-2 Data)
* **Deep Learning:** TensorFlow (CNN-BiLSTM)
* **Weights Management:** Git LFS / GitHub Desktop
* **Visualization:** Plotly & Leaflet

---

## 🚀 Deployment Status
The project is fully synchronized via GitHub Desktop. The weights files (`.h5`) are integrated into the root directory for seamless loading on Streamlit Cloud.

![Local Sync Status](assets/Screenshot_161.jpg)

---

## 🛠️ Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch: `streamlit run aravali_app.py`

---

## 🛡️ Conservation Impact
AFI provides environmentalists with a "Single Source of Truth." By tracking NDVI dips and AI classification, we can trigger early warnings for illegal mining or habitat loss before the damage becomes irreversible.

**Author:** [Vanshu Chaudhary](https://github.com/vanshuchaudhary)
