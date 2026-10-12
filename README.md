# Medical Insurance Cost Prediction

A beginner-friendly machine learning project that predicts medical insurance charges using customer demographic and lifestyle features.

**Student:** Nadeem Khan  
**Project Type:** Supervised Machine Learning – Regression  

---

## Project Objective

Predict individual medical insurance charges based on features such as age, BMI, smoking status, number of children, sex, and region. The project covers a complete end-to-end ML workflow: data cleaning → EDA → preprocessing → model training → evaluation → insights.

---

## Dataset

| Property        | Value                                      |
|-----------------|--------------------------------------------|
| File            | `Medical Insurance Cost Prediction.csv`    |
| Original rows   | 1,338                                      |
| After cleaning  | 1,337 (1 duplicate removed)                |
| Missing values  | None                                       |
| Target column   | `charges` (medical cost in USD)            |

**Features:**

| Column    | Type              | Description                          |
|-----------|-------------------|--------------------------------------|
| age       | Numerical (int)   | Age of insured person (18–64)        |
| sex       | Categorical       | Gender: male / female                |
| bmi       | Numerical (float) | Body Mass Index (15.96–53.13)        |
| children  | Numerical (int)   | Number of dependent children (0–5)   |
| smoker    | Categorical       | Smoking status: yes / no             |
| region    | Categorical       | US region: northeast/northwest/southeast/southwest |
| charges   | Numerical (float) | Medical insurance cost — **target**  |

---

## Project Structure

```
Medical Insurance Cost Prediction.csv   ← Original dataset (unchanged)
Medical_Insurance_Cost_Prediction.ipynb ← Main Jupyter notebook (all code)
Medical_Insurance_Cost_Prediction_Report.docx ← Project report (Word)
requirements.txt                        ← Python dependencies
README.md                               ← This file
model_results.txt                       ← Saved model metrics (auto-generated)
chart_*.png                             ← EDA and evaluation charts (auto-generated)
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the notebook

```bash
jupyter notebook Medical_Insurance_Cost_Prediction.ipynb
```

Or run all cells at once:

```bash
jupyter nbconvert --to notebook --execute --inplace Medical_Insurance_Cost_Prediction.ipynb
```

---

## Notebook Sections

| # | Section                            | Description                                      |
|---|------------------------------------|--------------------------------------------------|
| 1 | Import Libraries                   | pandas, numpy, matplotlib, seaborn, scikit-learn |
| 2 | Load and Understand the Dataset    | Shape, head, info, describe                      |
| 3 | Data Quality Check                 | Missing values, duplicates, range validation     |
| 4 | Exploratory Data Analysis (EDA)    | 7 charts covering distributions & relationships  |
| 5 | Feature Engineering & Preprocessing| Label encoding, one-hot encoding, scaling        |
| 6 | Model Training                     | Linear Regression & Random Forest Regressor      |
| 7 | Model Evaluation                   | MAE, RMSE, R² + 3 evaluation charts              |
| 8 | Results Summary                    | Side-by-side metric comparison table             |
| 9 | Business Insights & Conclusions    | Key findings and recommendations                 |

---

## Model Results (Actual)

| Model                     | MAE ($)   | RMSE ($)  | R² Score |
|---------------------------|-----------|-----------|----------|
| Linear Regression         | 4,177.05  | 5,956.34  | 0.8069   |
| **Random Forest Regressor** | **2,632.78** | **4,697.78** | **0.8799** |

**Best Model: Random Forest Regressor** — explains **87.99%** of the variance in insurance charges.

---

## Key Findings

1. **Smoking is the #1 predictor** — smokers pay ~3.8× more than non-smokers ($32,050 vs $8,441 average).
2. **Age** is the second most important feature (correlation = 0.298).
3. **BMI** matters especially when combined with smoking — highest charges occur in smoker + high-BMI cluster.
4. **Southeast region** has the highest average charges ($14,735), followed by Northeast ($13,406).
5. **Random Forest** outperforms Linear Regression on all three metrics by a significant margin.

---

## Charts Generated

| File                            | Description                             |
|---------------------------------|-----------------------------------------|
| chart_charges_distribution.png  | Raw and log-transformed charge histogram|
| chart_smoker_charges.png        | Boxplot: smoker vs non-smoker charges   |
| chart_age_vs_charges.png        | Scatter: age vs charges by smoker status|
| chart_bmi_vs_charges.png        | Scatter: BMI vs charges by smoker status|
| chart_region_charges.png        | Bar: average charges by region          |
| chart_correlation_heatmap.png   | Heatmap of feature correlations         |
| chart_children_charges.png      | Bar: average charges by children count  |
| chart_model_comparison.png      | MAE/RMSE/R² bar comparison              |
| chart_actual_vs_predicted.png   | Actual vs predicted scatter (both models)|
| chart_feature_importance.png    | Random Forest feature importance        |

---

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
nbconvert>=6.0.0
python-docx>=0.8.11
```

---

## Deliverables

- [x] `Medical_Insurance_Cost_Prediction.ipynb` — Jupyter notebook with all code and outputs
- [x] `requirements.txt` — Python package dependencies
- [x] `Medical_Insurance_Cost_Prediction_Report.docx` — Professional project report
- [x] `README.md` — This file

---

*Academic Internship Project — Machine Learning | Student: Nadeem Khan*
