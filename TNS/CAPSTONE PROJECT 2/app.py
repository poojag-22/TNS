# app.py
import streamlit as st
import requests
import json
import joblib
import pandas as pd

st.set_page_config(page_title='Heart Disease Predictor', layout='centered')
st.title('❤️ Heart Disease Predictor')
st.markdown("Use the sliders and dropdowns below. Short help text is shown for each field.")

# Load expected feature order (must match your JSON)
with open('feature_order.json') as f:
    FEATURES = json.load(f)

mode = st.selectbox('Mode', ['Call API (FastAPI)', 'Local model'])

# --- Inputs (explicit widgets so labels + ranges + help are clear) ---
st.subheader('Patient information & test values')

age = st.slider('age — Age (years)', min_value=1, max_value=100, value=55,
                help='Patient age in years. Typical adult range: 18–90.')
st.caption('Age: older age increases cardiovascular risk.')

sex_label = st.selectbox('sex — Sex', options=['Male', 'Female'], index=0,
                         help='Biological sex (used in many clinical scores).')
# Common encoding in heart datasets: Male=1, Female=0
sex = 1 if sex_label == 'Male' else 0
st.caption('sex mapping used here: Male = 1, Female = 0.')

chest_pain_map = {
    'Typical angina (1)': 1,
    'Atypical angina (2)': 2,
    'Non-anginal pain (3)': 3,
    'Asymptomatic (4)': 4
}
cp_choice = st.selectbox('chest_pain_type — Chest pain type', list(chest_pain_map.keys()))
chest_pain_type = chest_pain_map[cp_choice]
st.caption('Chest pain types: 1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic.')

resting_blood_pressure = st.slider('resting_blood_pressure — Resting BP (mm Hg)',
                                   min_value=80, max_value=220, value=120, step=1,
                                   help='Systolic blood pressure measured at rest.')
st.caption('Resting BP in mm Hg. Hypertension typically >130–140 mm Hg.')

cholesterol = st.slider('cholesterol — Serum cholesterol (mg/dl)',
                        min_value=100, max_value=600, value=200, step=1,
                        help='Total serum cholesterol (mg/dl).')
st.caption('Normal ~125–200 mg/dl; higher values increase risk.')

fbs_choice = st.selectbox('fasting_blood_sugar — Fasting blood sugar >120 mg/dl?',
                          ['No (0)', 'Yes (1)'])
fasting_blood_sugar = 1 if 'Yes' in fbs_choice else 0
st.caption('fasting_blood_sugar: 1 = >120 mg/dl (higher risk), 0 = ≤120 mg/dl.')

restecg_map = {
    'Normal (0)': 0,
    'ST-T wave abnormality (1)': 1,
    'Left ventricular hypertrophy (2)': 2
}
restecg_choice = st.selectbox('resting_ecg — Resting ECG result', list(restecg_map.keys()))
resting_ecg = restecg_map[restecg_choice]
st.caption('resting_ecg: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy.')

max_heart_rate = st.slider('max_heart_rate — Maximum heart rate achieved (bpm)',
                           min_value=60, max_value=220, value=150, step=1)
st.caption('Max HR during exercise test. Lower than expected peak HR may indicate ischemia.')

exang_choice = st.selectbox('exercise_induced_angina — Exercise-induced angina?', ['No (0)', 'Yes (1)'])
exercise_induced_angina = 1 if 'Yes' in exang_choice else 0
st.caption('exercise_induced_angina: 1 = angina during exercise, 0 = none.')

st_depression = st.slider('st_depression — ST depression induced by exercise relative to rest',
                          min_value=0.0, max_value=6.0, value=1.0, step=0.1,
                          help='Numeric ST depression (mm). Higher values can indicate ischemia.')
st.caption('ST depression (0.0–6.0): larger values often worse.')

st_slope_map = {
    'Upsloping (1)': 1,
    'Flat (2)': 2,
    'Downsloping (3)': 3
}
st_slope_choice = st.selectbox('st_slope — Slope of the ST segment at peak exercise', list(st_slope_map.keys()))
st_slope = st_slope_map[st_slope_choice]
st.caption('st_slope: 1=upsloping, 2=flat, 3=downsloping (downsloping worse).')

num_major_vessels = st.slider('num_major_vessels — Number of major vessels colored by fluoroscopy',
                              min_value=0, max_value=3, value=0, step=1)
st.caption('num_major_vessels (0–3): more vessels may indicate more severe disease.')

thal_map = {
    'Normal (3)': 3,
    'Fixed defect (6)': 6,
    'Reversible defect (7)': 7
}
thal_choice = st.selectbox('thalassemia — Thalassemia result', list(thal_map.keys()))
thalassemia = thal_map[thal_choice]
st.caption('thalassemia: common encoding 3=normal, 6=fixed defect, 7=reversible defect.')

# --- Assemble payload in same keys your model expects ---
payload = {
    'age': int(age),
    'sex': int(sex),
    'chest_pain_type': int(chest_pain_type),
    'resting_blood_pressure': int(resting_blood_pressure),
    'cholesterol': int(cholesterol),
    'fasting_blood_sugar': int(fasting_blood_sugar),
    'resting_ecg': int(resting_ecg),
    'max_heart_rate': int(max_heart_rate),
    'exercise_induced_angina': int(exercise_induced_angina),
    'st_depression': float(round(st_depression, 1)),
    'st_slope': int(st_slope),
    'num_major_vessels': int(num_major_vessels),
    'thalassemia': int(thalassemia)
}

#st.write('---')
#st.subheader('Preview of inputs')
#st.json(payload)

# --- Predict button ---
if st.button('🔍 Predict'):
    if mode == 'Call API (FastAPI)':
        try:
            resp = requests.post('http://127.0.0.1:8000/predict', json={'data': payload}, timeout=5)
            resp.raise_for_status()
            st.success(resp.json())
        except Exception as e:
            st.error('API call failed: ' + str(e))
    else:
        try:
            model = joblib.load('best_model.pkl')
            df = pd.DataFrame([payload])
            pred = int(model.predict(df)[0])
            proba = model.predict_proba(df)[0, 1] if hasattr(model, 'predict_proba') else None
            if proba is not None:
                st.success(f'Prediction: {pred}  —  Probability: {proba:.2f}')
            else:
                st.success(f'Prediction: {pred}')
        except Exception as e:
            st.error('Local model failed: ' + str(e))
