# 🌌 Exoplanet Analysis (Kepler & TOI Datasets)

This repository contains the **working Jupyter Notebook** for analyzing exoplanet data.  
The main file is:

👉 **`exoplanet_analysis.ipynb`** (this is the correct and tested version – the Kepler notebook).

Processed kepler datasets are stored in **`nasa/processed_data/`** and are required for the notebook to run properly.

---

## 📂 Repository Structure

## 📂 Repository Structure

```plaintext
├── nasa/
│   ├── processed_data/
│   │   ├── kepler_train_data.csv
│   │   ├── kepler_test_data_clean.csv
│   │   └── kepler_test_data.csv     # with labels
│   ├── toi-processed-data/
│   │   └── ... (TOI processed files)
│   ├── process_kepler_data.py       # preprocessing script for Kepler
│   ├── process_toi_data.py          # preprocessing script for TOI
│   ├── TOI_2025...csv               # raw datasets
│   ├── cumulative_2025...csv
│   └── k2pandc_2025...csv
│
├── exoplanet_analysis.ipynb         # MAIN working notebook
├── tess_toi_analysis.ipynb          # TOI/TESS analysis notebook
├── requirements.txt
└── README.md
