# Multi-Disease Risk Predictor

A Streamlit web app that estimates the risk of **heart disease**, **liver disease** and **diabetes** from clinical inputs, shows a confidence score for each prediction, and suggests relevant specialists when the risk is flagged.

Built as my B.Tech (Computer Engineering) major project at Silver Oak University, 2026.

> **Educational use only.** This is a student project, not a medical device, and it is not a substitute for professional medical advice.

**Live app:** https://ml-project-flxfm9v4qdvks6hqufmxhi.streamlit.app

<!-- Add a screenshot here: ![App screenshot](screenshot.png) -->

## What it does

1. You pick a disease tab (Heart, Liver or Diabetes) and enter clinical values in the form.
2. The matching model returns a probability. A per-disease threshold turns it into **Likely** or **Unlikely**.
3. The app shows the confidence score and a progress bar.
4. If the result is **Likely**, it lists up to 6 specialists, ranked by years of experience (highest first) and then consultation fee (lowest first).

## Models

| Disease | Model | Threshold | Why |
|---|---|---|---|
| Heart | Random Forest (200 trees, max depth 6, class-balanced) | 0.35 | Lower threshold favors recall, since a missed heart case costs more than a false alarm |
| Liver | Logistic Regression (class-balanced) | 0.65 | Higher threshold favors precision |
| Diabetes | Logistic Regression (class-balanced) | 0.65 | Higher threshold favors precision |

Preprocessing: drop ID columns, encode categorical values, fill missing numeric values with the column mean, then standardize with `StandardScaler`. Each model uses a stratified 80/20 train-test split with `random_state=42`.

## Results (held-out 20% test sets)

| Disease | Accuracy | Precision | Recall |
|---|---|---|---|
| Heart (Random Forest) | 0.81 | 0.76 | 0.91 |
| Liver (Logistic Regression) | 0.74 | 0.93 | 0.67 |
| Diabetes (Logistic Regression) | 0.73 | 0.60 | 0.70 |

The liver model is precise but misses about a third of true cases. The diabetes model has the weakest precision. Improving both is listed under future work.

## Datasets

| Dataset | Records | Features | Source |
|---|---|---|---|
| Heart Disease | 1,025 | 13 | UCI Heart Disease |
| Indian Liver Patient (ILPD) | 583 | 10 | UCI Machine Learning Repository |
| Pima Indians Diabetes | 768 | 8 | NIDDK / Kaggle |

## Doctor recommendations

The app looks for a doctor dataset named `doctors.csv`, `practo.csv` or `doctor.csv` in the project folder. It normalizes the columns (name, degree, speciality, city, location, fee, experience) and matches each disease to specialities by keyword:

- Heart: cardiology, cardiac, cardiothoracic
- Liver: hepatology, gastroenterology
- Diabetes: endocrinology

If no doctor file is found, the app shows three sample entries so the feature can still be demonstrated. The Practo-style dataset used during development is not included in this repository.

## Run locally

```bash
git clone https://github.com/Vishwa-1033/ML-project.git
cd ML-project
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py            Streamlit app: training, prediction, doctor recommendation
heart.csv         Heart disease dataset
liver.csv         Liver patient dataset
diabetes.csv      Diabetes dataset
requirements.txt  Python dependencies
```

## Testing

I ran 49 test cases (18 unit, 7 integration, 10 system, 6 performance, 8 edge cases) and all passed. Average inference time for the full pipeline was about 189 ms. Models are trained once at startup and cached, so the first page load takes longer.

## Limitations

- The datasets are small, and the liver and diabetes sets are class-imbalanced.
- Models are trained inside the app at startup rather than saved to disk.
- Results come from public benchmark datasets and have not been validated on real patients.

## Future work

- Try XGBoost or TabNet, and SMOTE for the imbalanced liver data
- Add SHAP explanations for each prediction
- Filter doctor recommendations by location
- Save trained models and add an evaluation script that prints the metrics

## Author

**Vishwa Panchal**, B.Tech Computer Engineering, Silver Oak University
Email: vishwap1212@gmail.com
