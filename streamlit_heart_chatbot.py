import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit settings
st.set_page_config(page_title="Heart Disease Risk Chatbot", page_icon="🫀")
st.title("🫀 Heart Disease Risk Chatbot (Chat-Style)")
st.markdown("Answer the following questions one by one to assess your heart disease risk.")

# Field order and questions
questions = [
    ("age", "What is your age?"),
    ("sex", "What is your biological sex? (0 = Female, 1 = Male)"),
    ("cp", "Chest pain type (0–3)?"),
    ("trestbps", "Resting blood pressure (mm Hg)?"),
    ("chol", "Cholesterol level (mg/dl)?"),
    ("fbs", "Fasting blood sugar > 120 mg/dl? (0 = No, 1 = Yes)"),
    ("restecg", "Resting ECG result (0 = Normal, 1 = ST-T abnormality, 2 = LVH)?"),
    ("thalach", "Maximum heart rate achieved?"),
    ("exang", "Exercise-induced angina? (0 = No, 1 = Yes)"),
    ("oldpeak", "ST depression induced by exercise?"),
    ("slope", "Slope of the peak exercise ST segment (0–2)?"),
    ("ca", "Number of major vessels coloured by fluoroscopy (0–4)?"),
    ("thal", "Thalassemia (0 = Normal, 1 = Fixed defect, 2 = Reversible defect)?")
]

# PDF generator
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
        "ca": "Major Vessels Colored (0–4)",
        "thal": "Thalassemia (0=Normal, 1=Fixed, 2=Reversible)"
    }

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)
    text.textLine("Heart Disease Risk Assessment Report")
    text.textLine("--------------------------------------")
    for k, v in input_data.items():
        label = field_names.get(k, k)
        text.textLine(f"{label}: {v}")
    text.textLine(f"\nPredicted Risk: {round(prediction, 2)}%")
    text.textLine(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Streamlit session state
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.inputs = {}

# Display chatbot question
if st.session_state.step < len(questions):
    _, q_text = questions[st.session_state.step]
    user_input = st.chat_input(q_text)
    if user_input:
        key = questions[st.session_state.step][0]
        try:
            value = float(user_input) if "." in user_input or key == "oldpeak" else int(user_input)
            st.session_state.inputs[key] = value
            st.session_state.step += 1
        except ValueError:
            st.error("Please enter a valid number.")

# When all data is collected
if st.session_state.step == len(questions):
    input_order = [k for k, _ in questions]
    input_list = [st.session_state.inputs[k] for k in input_order]
    input_array = scaler.transform([input_list])
    prediction = model.predict_proba(input_array)[0][1] * 100

    st.success(f"🧠 Your predicted heart disease risk is **{round(prediction, 2)}%**.")
    if prediction > 70:
        st.warning("⚠️ This is a high risk. Please consult a medical professional.")
    elif prediction > 40:
        st.info("🔍 This is a moderate risk. A check-up is recommended.")
    else:
        st.info("✅ This appears to be a low risk. Keep up the healthy lifestyle!")

    # Pie chart
    labels = ['At Risk', 'No Risk']
    sizes = [round(prediction, 2), 100 - round(prediction, 2)]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=['red', 'green'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.markdown("### 🧩 Risk Distribution")
    st.pyplot(fig)

    # PDF download
    pdf = generate_pdf(st.session_state.inputs, prediction)
    st.download_button("📄 Download PDF Report", data=pdf, file_name="heart_risk_report.pdf", mime="application/pdf")

    # Reset session for new user
    st.session_state.step = 0
    st.session_state.inputs = {}
