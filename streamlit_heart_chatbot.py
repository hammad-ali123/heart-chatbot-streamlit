
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

CSV_FILE = "prediction_history.csv"

model = joblib.load("lr_model.pkl")
scaler = joblib.load("lr_scaler.pkl")

st.title("🫀 Heart Disease Risk Chatbot (Logistic Regression)")
st.markdown("Answer the questions below to assess your heart disease risk.")

age = st.number_input("Age", min_value=1, max_value=120, value=45, key="age")
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="sex")
cp = st.selectbox("Chest pain type (0–3)", [0, 1, 2, 3], key="cp")
trestbps = st.number_input("Resting blood pressure", min_value=80, max_value=200, value=120, key="trestbps")
chol = st.number_input("Cholesterol level", min_value=100, max_value=600, value=240, key="chol")
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", [0, 1], key="fbs")
restecg = st.selectbox("Resting ECG results (0–2)", [0, 1, 2], key="restecg")
thalach = st.number_input("Maximum heart rate achieved", min_value=60, max_value=250, value=150, key="thalach")
exang = st.selectbox("Exercise-induced angina", [0, 1], key="exang")
oldpeak = st.number_input("ST depression induced by exercise", value=1.0, step=0.1, key="oldpeak")
slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2], key="slope")
ca = st.selectbox("Number of major vessels (0–4) coloured by fluoroscopy", [0, 1, 2, 3, 4], key="ca")
thal = st.selectbox("Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)", [0, 1, 2], key="thal")

def save_prediction(data, prediction):
    data["prediction (%)"] = round(prediction, 2)
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, mode='w', header=True, index=False)

def generate_pdf(input_data, prediction):
    field_names = {
        "age": "Age",
        "sex": "Sex (0=Female, 1=Male)",
        "cp": "Chest Pain Type (0–3)",
        "trestbps": "Resting Blood Pressure (mm Hg)",
        "chol": "Cholesterol Level (mg/dl)",
        "fbs": "Fasting Blood Sugar > 120 (0=No, 1=Yes)",
        "restecg": "Resting ECG Result (0–2)",
        "thalach": "Max Heart Rate Achieved",
        "exang": "Exercise-Induced Angina (0=No, 1=Yes)",
        "oldpeak": "ST Depression by Exercise",
        "slope": "Slope of Peak Exercise ST",
        "ca": "Major Vessels Coloured (0–4)",
        "thal": "Thalassemia (0=Normal, 1=Fixed, 2=Reversible)"
    }
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)
    text.textLine("Heart Disease Risk Assessment Report")
    text.textLine("--------------------------------------")
    for key, value in input_data.items():
        label = field_names.get(key, key)
        text.textLine(f"{label}: {value}")
    text.textLine(f"Predicted Risk: {round(prediction, 2)}%")
    text.textLine(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

if st.button("Check Risk"):
    try:
        inputs = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]
        X = scaler.transform([inputs])
        prediction = model.predict_proba(X)[0][1] * 100
        input_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        save_prediction(input_data, prediction)

        st.success(f"🧠 Predicted heart disease risk: **{round(prediction, 2)}%**")
        if prediction > 70:
            st.warning("⚠️ High risk! Please consult a medical professional.")
        elif prediction > 40:
            st.info("🔍 Moderate risk. A medical check-up is recommended.")
        else:
            st.info("✅ Low risk. Keep maintaining a healthy lifestyle!")

        labels = ['At Risk', 'No Risk']
        sizes = [round(prediction, 2), 100 - round(prediction, 2)]
        colors = ['red', 'green']
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.markdown("### 🧩 Risk Distribution")
        st.pyplot(fig)

        pdf = generate_pdf(input_data, prediction)
        st.download_button("📄 Download PDF Report", data=pdf,
                           file_name="heart_risk_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")

if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "rb") as f:
        st.download_button("📥 Download All Predictions", f, file_name=CSV_FILE)
