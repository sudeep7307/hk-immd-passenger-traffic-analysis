# 🛃 Hong Kong Immigration Passenger Traffic Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Issues](https://img.shields.io/github/issues/solarlaziers/hk-immd-passenger-traffic-analysis)](https://github.com/YOUR-USERNAME/hk-immd-passenger-traffic-analysis/issues)

A comprehensive data science project analyzing daily passenger traffic at Hong Kong immigration checkpoints using machine learning algorithms. This project was developed for the Data Science course at THEi FDE (Semester 1 AY2025/26).

## 📊 Project Overview

This project leverages open data from the Hong Kong Immigration Department to:
- **Predict** future passenger traffic using Linear Regression
- **Classify** high vs. low traffic days using Logistic Regression and SVM
- **Cluster** similar traffic patterns using K-means
- **Provide actionable insights** for immigration resource planning

## 🎯 Objectives

1. Analyze historical passenger traffic patterns
2. Build predictive models for traffic forecasting
3. Identify peak traffic periods and seasonal trends
4. Develop classification systems for operational planning
5. Generate visualizations for trend analysis

## 📁 Dataset

**Source:** [HK ImmD SET5 – Statistics on Daily Passenger Traffic](https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic/resource/e06a2a45-fe05-4eb4-9302-237d74343d52)

**Features Include:**
- `date`: Date of recording
- `immigration_point`: Specific checkpoint location
- `passenger_type`: Type of passenger (resident, visitor, etc.)
- `traffic_count`: Number of passengers
- Additional metadata as available

## 🧠 Algorithms Implemented

| Algorithm | Purpose | Library Used |
|-----------|---------|--------------|
| **Linear Regression** | Traffic volume prediction | `sklearn.linear_model` |
| **Logistic Regression** | Binary classification (High/Low traffic) | `sklearn.linear_model` |
| **Support Vector Machine (SVM)** | Non-linear classification | `sklearn.svm` |
| **K-means Clustering** | Pattern discovery and grouping | `sklearn.cluster` |

## 🛠️ Tech Stack

- **Programming Language:** Python 3.8+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Development:** Jupyter Notebook
- **Version Control:** Git & GitHub

## 📂 Project Structure

hk-immd-passenger-traffic-analysis/
├── data/                  # Data directory
│   ├── raw/               # Original, immutable data (e.g., CSV files from IMMD)
│   └── processed/         # Cleaned and transformed data (e.g., merged datasets with features)
├── notebooks/             # Jupyter notebooks
│   └── main_analysis.ipynb # Main analysis notebook (EDA, models, insights)
├── scripts/               # Python scripts
│   ├── data_preprocessing.py # Data cleaning and feature engineering functions
│   ├── models.py          # ML model implementations (Linear Regression, K-Means, SVM)
│   └── visualization.py   # Plotting and visualization functions
├── reports/               # Reports and presentations
│   ├── presentation.pptx  # PowerPoint presentation (20 slides max, as per project guidelines)
│   └── final_report.pdf   # Final project report (PDF exported from Jupyter with >1000 words explanations)
├── .gitignore             # Git ignore file (excludes temp files, environments, etc.)
├── requirements.txt       # Project dependencies (pip-installable list)
├── README.md              # This file (project overview and documentation)
└── LICENSE                # MIT License

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- pip (Python package manager)

### Installation

1. **Clone the repository:**

```bash
   git clone https://github.com/YOUR-USERNAME/hk-immd-passenger-traffic-analysis.git
   cd hk-immd-passenger-traffic-analysis
```
2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies:**