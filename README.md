# 🌲 Aravali Forest Intelligence (AFI) 🛰️
### **Real-Time AI-Powered Satellite Surveillance for the Aravalli Range**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://vanshuchaudhary-aravali-sentinel-aravali-app-fmcsqj.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

The Aravalli Range is one of the world's oldest mountain ranges, acting as a critical green barrier against desertification. **AFI** leverages Deep Learning (CNN-BiLSTM) and Google Earth Engine (GEE) to provide real-time ecological health monitoring, trend forecasting, and risk assessment.

---

## 🌟 Key Features
* **🤖 Precision AI Analysis:** Custom-trained neural networks provide site-specific classification with confidence scores up to **92%+**.
* **📈 7-Year Temporal Tracking:** Direct satellite measurement of forest health trends (NDVI) from 2020 to 2026.
* **🗺️ Risk Hotspot Mapping:** Identification of degraded and encroached forest zones using multi-spectral imagery.
* **🛠️ Restoration Intelligence:** Automated recommendations for native species (like Dhau) and biodiversity recovery.

---

## 📸 Project Showcase

### **1. AI-Powered Monitoring Dashboard**
The engine identifies forest status at specific coordinates. In this example, the AI detects a degraded zone with high precision, highlighting topsoil erosion and corridor fragmentation risks.
![AI Analysis](screenshots/Screenshot%20(164).png)

### **2. 7-Year Ecological Health Trends**
Tracking the "heartbeat" of the forest. This graph uses direct satellite measurements to visualize how vegetation health has fluctuated over nearly a decade.
![Temporal Trends](screenshots/Screenshot%20(165).png)

### **3. Smart Restoration Strategies**
The system doesn't just find problems; it suggests solutions. From recommending **Native Dhau** plantation to identifying invasive species, it provides an actionable roadmap for conservation.
![Restoration Strategy](screenshots/Screenshot%20(166).png)

---

## 🏗️ Project Structure
```text
Aravali-Sentinel/
├── aravali_app.py              # Main Streamlit Application logic
├── aravali_forest_model_v2.weights.h5 # Deep Learning Weights (Primary)
├── aravli_classifier_v2.weights.h5    # Secondary Classifier Weights
├── requirements.txt            # Python Dependencies
├── screenshot/                 # Project Visuals & Assets
│   ├── Screenshot (161).jpg
│   ├── Screenshot (164).jpg
│   ├── Screenshot (165).png
│   └── Screenshot (166).jpg
├── .gitattributes              # Git LFS configuration for large files
└── README.md                   # Project Documentation
```
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

![Local Sync Status](screenshots/Screenshot%20(161).png)

---

## 🛠️ Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch: `streamlit run aravali_app.py`

---

## 🛡️ Conservation Impact
AFI provides environmentalists with a "Single Source of Truth." By tracking NDVI dips and AI classification, we can trigger early warnings for illegal mining or habitat loss before the damage becomes irreversible.

**Author:** [Vanshu Chaudhary](https://github.com/vanshuchaudhary)
