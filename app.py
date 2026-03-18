import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Disease Risk Predictor",
    page_icon="",
    layout="centered"
)

st.title("Disease Risk Predictor")
st.markdown("Enter patient details below to assess risk for Heart Disease, Liver Disease, and Diabetes.")

@st.cache_data
def load_doctor_data():
    doctor_df   = None
    hospital_df = None
    for fname in ["doctors.csv", "practo.csv", "doctor.csv"]:
        try:
            doctor_df = pd.read_csv(fname, low_memory=False)
            break
        except:
            pass
    if doctor_df is None:
        for fname in ["doctors.json", "doctor.json"]:
            try:
                doctor_df = pd.read_json(fname, lines=True)
                break
            except:
                try:
                    doctor_df = pd.read_json(fname, orient='records')
                    break
                except:
                    pass
    for fname in ["hospital.csv", "hospitals.csv", "treatment.csv"]:
        try:
            hospital_df = pd.read_csv(fname, low_memory=False)
            break
        except:
            pass
    if doctor_df is None and hospital_df is None:
        doctor_df = pd.DataFrame({
            "Name":                ["Dr. A Mehta",    "Dr. P Sharma",   "Dr. R Patel"],
            "Degree":              ["MBBS, MD",        "MBBS, DM",       "MBBS, MS"],
            "Speciality":          ["Cardiologist",    "Endocrinologist", "Hepatologist"],
            "Location":            ["City Hospital",   "Care Clinic",    "Liver Institute"],
            "City":                ["Mumbai",          "Delhi",          "Ahmedabad"],
            "Consult Fee":         [800,               600,              700],
            "Years of Experience": [12,                10,               15]
        })
    return doctor_df, hospital_df

def normalize_doctor_df(df):
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    colmap = {}
    for c in df.columns:
        low = c.lower()
        if ("name" in low and "doctor" in low) or low.strip() == "name":
            colmap[c] = "Name"
        elif "degree" in low or "qualification" in low:
            colmap[c] = "Degree"
        elif "special" in low or "speciality" in low or "specialization" in low:
            colmap[c] = "Speciality"
        elif "city" in low:
            colmap[c] = "City"
        elif "loc" in low or "hospital" in low or "address" in low:
            colmap[c] = "Location"
        elif "fee" in low or "consult" in low:
            colmap[c] = "Consult Fee"
        elif "experience" in low or "exp" in low:
            colmap[c] = "Years of Experience"
    df = df.rename(columns=colmap)
    for expected in ["Name", "Degree", "Speciality", "Location",
                     "City", "Consult Fee", "Years of Experience"]:
        if expected not in df.columns:
            df[expected] = pd.NA
    df["Consult Fee"] = (
        df["Consult Fee"].astype(str)
        .apply(lambda v: re.sub(r"[^\d\.]", "", v) if pd.notna(v) else "")
    )
    df["Consult Fee"] = pd.to_numeric(
        df["Consult Fee"].replace("", np.nan), errors="coerce"
    ).fillna(0.0)
    df["Years of Experience"] = (
        df["Years of Experience"].astype(str)
        .apply(lambda v: re.findall(r"\d+", v)[0] if re.findall(r"\d+", v) else "")
    )
    df["Years of Experience"] = pd.to_numeric(
        df["Years of Experience"].replace("", np.nan), errors="coerce"
    ).fillna(0.0)
    df["Speciality_clean"] = (
        df["Speciality"].astype(str)
        .str.lower()
        .str.replace(r"[^a-z, ]", "", regex=True)
        .str.strip()
    )
    df = df.assign(
        Speciality_clean=df["Speciality_clean"].str.split(",")
    ).explode("Speciality_clean")
    df["Speciality_clean"] = df["Speciality_clean"].astype(str).str.strip()
    df = df[df["Speciality_clean"].notna() & (df["Speciality_clean"] != "")]
    return df

disease_to_speciality = {
    "Heart Disease": [
        "cardio", "cardiologist", "cardiac", "heart",
        "interventional cardiology", "cardiothoracic", "cardiovascular"
    ],
    "Liver Disease": [
        "hepatol", "hepato", "hepatic", "gastro",
        "gastroenterologist", "liver", "hepatologist"
    ],
    "Diabetes": [
        "endocrin", "diabet", "diabetes", "endocrinologist"
    ]
}

def recommend_doctors(detected_diseases, doctor_df_clean,
                      hospital_df_clean, top_n=6):
    frames = []
    for disease in detected_diseases:
        kws = disease_to_speciality.get(disease, [])
        if not kws:
            continue
        if not doctor_df_clean.empty:
            mask  = doctor_df_clean["Speciality_clean"].apply(
                lambda s: any(k in s for k in kws))
            found = doctor_df_clean[mask].copy()
            if not found.empty:
                found["Source"]      = "Practo"
                found["Matched_for"] = disease
                frames.append(found)
        if not hospital_df_clean.empty:
            mask  = hospital_df_clean["Speciality_clean"].apply(
                lambda s: any(k in s for k in kws))
            found = hospital_df_clean[mask].copy()
            if not found.empty:
                found["Source"]      = "Hospital"
                found["Matched_for"] = disease
                frames.append(found)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        by=["Years of Experience", "Consult Fee"],
        ascending=[False, True]
    )
    out = out.drop_duplicates(subset=["Name", "Matched_for"])
    return out.head(top_n)

def show_doctor_recommendations(detected_diseases,
                                 doctor_df_clean, hospital_df_clean):
    st.divider()
    st.subheader("Recommended Doctors")
    recs = recommend_doctors(detected_diseases, doctor_df_clean,
                              hospital_df_clean, top_n=6)
    if recs.empty:
        st.info("No matching doctors found in the dataset.")
        return
    display_cols = ["Name", "Degree", "Speciality", "City",
                    "Location", "Consult Fee",
                    "Years of Experience", "Matched_for"]
    available = [c for c in display_cols if c in recs.columns]
    for _, row in recs[available].iterrows():
        with st.container():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{row.get('Name','N/A')}**")
                st.caption(f"{row.get('Degree','N/A')} - {row.get('Speciality','N/A')}")
                st.caption(f"Location: {row.get('Location','N/A')}, {row.get('City','N/A')}")
            with col2:
                fee = row.get('Consult Fee', 0)
                exp = row.get('Years of Experience', 0)
                st.metric("Fee", f"Rs {int(fee)}")
                st.metric("Experience", f"{int(exp)} yrs")
            st.caption(f"Recommended for: {row.get('Matched_for','N/A')}")
            st.divider()

@st.cache_resource
def load_and_train():
    heart_df    = pd.read_csv("heart.csv")
    liver_df    = pd.read_csv("liver.csv")
    diabetes_df = pd.read_csv("diabetes.csv")

    def preprocess_dataset(df, target_col, categorical_mapping={}):
        df = df.copy()
        df = df.drop(columns=['id'], errors='ignore')
        for col in df.select_dtypes(include='object').columns:
            if col in categorical_mapping:
                df[col] = df[col].map(categorical_mapping[col]).fillna(0)
            else:
                df[col] = df[col].map(
                    {'yes':1,'no':0,'M':1,'F':0,'Male':1,'Female':0}
                ).fillna(df[col])
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        X = df.drop(columns=[target_col])
        Y = df[target_col]
        return X, Y

    X_heart, Y_heart = preprocess_dataset(heart_df, 'target')
    liver_map = {'Gender': {'Male': 1, 'Female': 0}}
    X_liver, Y_liver = preprocess_dataset(liver_df, 'Dataset',
                                           categorical_mapping=liver_map)
    unique_vals = sorted(liver_df['Dataset'].dropna().unique())
    if set(unique_vals) == {1, 2}:
        Y_liver = Y_liver.apply(lambda x: 0 if x == 2 else int(x))
    else:
        Y_liver = Y_liver.astype(int)
    X_diabetes, Y_diabetes = preprocess_dataset(diabetes_df, 'Outcome')

    def train_heart(X, Y):
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y, test_size=0.2, stratify=Y, random_state=42)
        model = RandomForestClassifier(
            n_estimators=200, max_depth=6,
            class_weight='balanced', random_state=42)
        model.fit(X_train, Y_train)
        return model, scaler

    def train_lr(X, Y):
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y, test_size=0.2, stratify=Y, random_state=42)
        model = LogisticRegression(max_iter=5000, class_weight='balanced')
        model.fit(X_train, Y_train)
        return model, scaler

    heart_model,    heart_scaler    = train_heart(X_heart,    Y_heart)
    liver_model,    liver_scaler    = train_lr(X_liver,    Y_liver)
    diabetes_model, diabetes_scaler = train_lr(X_diabetes, Y_diabetes)

    return (
        heart_model,    heart_scaler,
        liver_model,    liver_scaler,
        diabetes_model, diabetes_scaler
    )

with st.spinner("Loading and training models..."):
    (heart_model,    heart_scaler,
     liver_model,    liver_scaler,
     diabetes_model, diabetes_scaler) = load_and_train()

doctor_df_raw,   hospital_df_raw   = load_doctor_data()
doctor_df_clean  = normalize_doctor_df(doctor_df_raw)
hospital_df_clean = normalize_doctor_df(hospital_df_raw)

st.success("Models ready!")

HEART_THRESHOLD    = 0.35
LIVER_THRESHOLD    = 0.65
DIABETES_THRESHOLD = 0.65

tab1, tab2, tab3 = st.tabs(["Heart Disease", "Liver Disease", "Diabetes"])

with tab1:
    st.subheader("Heart Disease Risk Assessment")
    col1, col2, col3 = st.columns(3)
    with col1:
        age      = st.number_input("Age",             min_value=1,   max_value=120, value=50,  key="h_age")
        trestbps = st.number_input("Resting BP",      min_value=80,  max_value=200, value=120, key="h_bp")
        fbs      = st.selectbox("Fasting BS > 120",   [0, 1],                        key="h_fbs")
        exang    = st.selectbox("Exercise Angina",    [0, 1],                        key="h_exang")
        ca       = st.selectbox("Major Vessels (0-3)",[0, 1, 2, 3],                 key="h_ca")
    with col2:
        sex      = st.selectbox("Sex", [0, 1],
                    format_func=lambda x: "Female" if x==0 else "Male",              key="h_sex")
        chol     = st.number_input("Cholesterol",     min_value=100, max_value=600, value=200, key="h_chol")
        restecg  = st.selectbox("Resting ECG",        [0, 1, 2],                    key="h_ecg")
        oldpeak  = st.number_input("Oldpeak",         min_value=0.0, max_value=10.0,value=1.0,
                                    step=0.1,                                         key="h_op")
        thal     = st.selectbox("Thal",               [0, 1, 2, 3],                 key="h_thal")
    with col3:
        cp       = st.selectbox("Chest Pain Type",    [0, 1, 2, 3, 4],              key="h_cp")
        thalach  = st.number_input("Max Heart Rate",  min_value=60,  max_value=250, value=150, key="h_hr")
        slope    = st.selectbox("Slope",              [0, 1, 2],                    key="h_slope")

    if st.button("Predict Heart Disease Risk", use_container_width=True):
        values = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]
        arr    = np.array(values, dtype=float).reshape(1, -1)
        prob   = heart_model.predict_proba(heart_scaler.transform(arr))[0][1]
        result = "Likely" if prob >= HEART_THRESHOLD else "Unlikely"
        st.divider()
        if result == "Likely":
            st.error(f"Heart Disease Risk: {result}")
            show_doctor_recommendations(
                ["Heart Disease"], doctor_df_clean, hospital_df_clean)
        else:
            st.success(f"Heart Disease Risk: {result}")
        st.metric("Model Confidence", f"{prob:.0%}")
        st.progress(float(prob))

with tab2:
    st.subheader("Liver Disease Risk Assessment")
    col1, col2 = st.columns(2)
    with col1:
        l_age    = st.number_input("Age",                       min_value=1,   max_value=120,  value=40,  key="l_age")
        l_gender = st.selectbox("Gender", [1, 0],
                    format_func=lambda x: "Male" if x==1 else "Female",                        key="l_gender")
        l_tb     = st.number_input("Total Bilirubin",           min_value=0.0, max_value=75.0, value=1.0,
                                    step=0.1,                                                   key="l_tb")
        l_db     = st.number_input("Direct Bilirubin",          min_value=0.0, max_value=20.0, value=0.3,
                                    step=0.1,                                                   key="l_db")
        l_ap     = st.number_input("Alkaline Phosphotase",      min_value=0,   max_value=2000, value=200, key="l_ap")
    with col2:
        l_aa     = st.number_input("Alamine Aminotransferase",  min_value=0,   max_value=2000, value=35,  key="l_aa")
        l_as_val = st.number_input("Aspartate Aminotransferase",min_value=0,   max_value=5000, value=40,  key="l_as")
        l_tp     = st.number_input("Total Proteins",            min_value=0.0, max_value=10.0, value=6.5,
                                    step=0.1,                                                   key="l_tp")
        l_alb    = st.number_input("Albumin",                   min_value=0.0, max_value=6.0,  value=3.5,
                                    step=0.1,                                                   key="l_alb")
        l_agr    = st.number_input("Albumin/Globulin Ratio",    min_value=0.0, max_value=3.0,  value=1.0,
                                    step=0.1,                                                   key="l_agr")

    if st.button("Predict Liver Disease Risk", use_container_width=True):
        values = [l_age, l_gender, l_tb, l_db, l_ap,
                  l_aa, l_as_val, l_tp, l_alb, l_agr]
        arr    = np.array(values, dtype=float).reshape(1, -1)
        prob   = liver_model.predict_proba(liver_scaler.transform(arr))[0][1]
        result = "Likely" if prob >= LIVER_THRESHOLD else "Unlikely"
        st.divider()
        if result == "Likely":
            st.error(f"Liver Disease Risk: {result}")
            show_doctor_recommendations(
                ["Liver Disease"], doctor_df_clean, hospital_df_clean)
        else:
            st.success(f"Liver Disease Risk: {result}")
        st.metric("Model Confidence", f"{prob:.0%}")
        st.progress(float(prob))

with tab3:
    st.subheader("Diabetes Risk Assessment")
    col1, col2 = st.columns(2)
    with col1:
        d_preg    = st.number_input("Pregnancies",       min_value=0,   max_value=20,   value=1,   key="d_preg")
        d_glucose = st.number_input("Glucose",           min_value=0,   max_value=300,  value=100, key="d_gluc")
        d_bp      = st.number_input("Blood Pressure",    min_value=0,   max_value=150,  value=70,  key="d_bp")
        d_skin    = st.number_input("Skin Thickness",    min_value=0,   max_value=100,  value=20,  key="d_skin")
    with col2:
        d_insulin = st.number_input("Insulin",           min_value=0,   max_value=900,  value=80,  key="d_ins")
        d_bmi     = st.number_input("BMI",               min_value=0.0, max_value=70.0, value=25.0,
                                     step=0.1,                                                      key="d_bmi")
        d_dpf     = st.number_input("Diabetes Pedigree", min_value=0.0, max_value=3.0,  value=0.5,
                                     step=0.01,                                                     key="d_dpf")
        d_age     = st.number_input("Age",               min_value=1,   max_value=120,  value=30,  key="d_age")

    if st.button("Predict Diabetes Risk", use_container_width=True):
        values = [d_preg, d_glucose, d_bp, d_skin,
                  d_insulin, d_bmi, d_dpf, d_age]
        arr    = np.array(values, dtype=float).reshape(1, -1)
        prob   = diabetes_model.predict_proba(diabetes_scaler.transform(arr))[0][1]
        result = "Likely" if prob >= DIABETES_THRESHOLD else "Unlikely"
        st.divider()
        if result == "Likely":
            st.error(f"Diabetes Risk: {result}")
            show_doctor_recommendations(
                ["Diabetes"], doctor_df_clean, hospital_df_clean)
        else:
            st.success(f"Diabetes Risk: {result}")
        st.metric("Model Confidence", f"{prob:.0%}")
        st.progress(float(prob))

st.divider()
st.caption("This tool is for educational purposes only and is not a substitute for professional medical advice.")
