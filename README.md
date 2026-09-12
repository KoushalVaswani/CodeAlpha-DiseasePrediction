# ❤️ Disease Prediction — Heart Disease

**Predicting heart disease presence from clinical data** — built for the CodeAlpha ML Internship

---

## 🎯 Objective
Predict whether a patient has heart disease based on clinical measurements
and test results — age, chest pain type, blood pressure, cholesterol,
ECG results, and more.

## 📊 Dataset
**UCI Heart Disease (Cleveland)** — 303 patients × 13 features
Source: https://archive.ics.uci.edu/dataset/45/heart+disease

| | |
|---|---|
| 👤 Demographics | age, sex |
| 🩺 Clinical measurements | resting BP, cholesterol, max heart rate |
| 🔬 Test results | chest pain type, ECG, thallium stress test, fluoroscopy |
| 🎯 Target | Disease present (46%) vs. absent (54%) |

## 🛠️ Approach
1. **EDA** — explored feature distributions and their relationship to disease
   presence; found max heart rate and chest pain type as visually strong signals.
2. **Preprocessing** — imputed 6 missing values (median), one-hot encoded
   nominal categorical features, scaled numeric features, stratified 80/20 split.
3. **Model Comparison** — Logistic Regression, SVM, Random Forest, XGBoost.
4. **Tuning** — GridSearchCV (5-fold stratified CV) on the best model, Random Forest.

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.869 | 0.812 | 0.929 | 0.867 | 0.951 |
| SVM | 0.820 | 0.774 | 0.857 | 0.814 | 0.920 |
| Random Forest (default) | 0.902 | 0.844 | 0.964 | 0.900 | 0.951 |
| XGBoost | 0.852 | 0.788 | 0.929 | 0.852 | 0.919 |
| **Random Forest (Tuned)** | **0.902** | **0.867** | 0.929 | **0.897** | **0.962** 🏆 |

## 💡 Key Findings

**🏆 Best model:** Tuned Random Forest — highest accuracy, F1, and ROC-AUC
(0.962), with strong recall (92.9%), meaning it correctly identifies the
vast majority of actual disease cases.

**🔍 Diagnostic tests beat general risk factors:** feature importance
ranked `thal` (thallium stress test), `cp` (chest pain type), and `ca`
(blocked vessels) as the top predictors — ahead of commonly cited risk
factors like `age`, `cholesterol`, and `blood pressure`, which ranked
lowest. This suggests specific diagnostic test results carry more
predictive signal than general demographic risk factors alone.

**⚖️ Different dataset, different winner:** unlike the Credit Scoring
project (where plain Logistic Regression won), Random Forest was the
clear best performer here — a reminder that model choice should always
be validated per-dataset, not assumed from a previous project.

## 🧰 Tech Stack
`Python` · `pandas` · `scikit-learn` · `xgboost` · `matplotlib` · `seaborn`

## 🚀 How to Run
```bash
git clone https://github.com/KoushalVaswani/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction
pip install -r requirements.txt
jupyter notebook notebooks/Heart_Disease_EDA_Baseline.ipynb
```

---
📩 Part of the **CodeAlpha Machine Learning Internship**
