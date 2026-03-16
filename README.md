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

### **1. AI-Powered Monitoring**
The system identifies forest status at specific coordinates. Below, the AI detects a degraded zone with **92.96% confidence**, highlighting fragmentation risks.
![AI Analysis](<img width="1920" height="1080" alt="Screenshot (164)" src="https://github.com/user-attachments/assets/f085771b-1d40-48be-bcf5-8d03dd88c840" />
)

### **2. Ecological Health Trends**
A direct look at the 7-year "heartbeat" of the forest. The graph tracks the Health Index, allowing users to see exactly when degradation occurred.
![Temporal Trends](<img width="1920" height="1080" alt="Screenshot (165)" src="https://github.com/user-attachments/assets/bf0699d9-14fa-448c-bab9-13f3167de9ed" />
)

### **3. Restoration & Actionable Insights**
Beyond monitoring, the app suggests specific strategies like "Native Dhau plantation" to restore biodiversity corridors.
![Restoration Strategy](<img width="1920" height="1080" alt="Screenshot (166)" src="https://github.com/user-attachments/assets/71279540-8535-4320-b082-2d675f9e4d92" />
)

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
