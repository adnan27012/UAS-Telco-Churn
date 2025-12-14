import streamlit as st
import pandas as pd
import joblib

# 1. Load Model
try:
    model = joblib.load('model_churn_terbaik.pkl')
except FileNotFoundError:
    st.error("File model tidak ditemukan! Pastikan 'model_churn_terbaik.pkl' ada satu folder dengan app.py")
    st.stop()

# 2. Judul Web
st.set_page_config(page_title="Prediksi Telco Churn", layout="wide")
st.title(" Aplikasi Prediksi Churn Pelanggan")
st.markdown("---")

# 3. Sidebar Input
st.sidebar.header(" Masukkan Data Pelanggan")

def user_input_features():
    # -- Demografi --
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

    # -- Layanan --
    tenure = st.sidebar.slider("Lama Langganan (Bulan)", 0, 72, 12)
    phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multi_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    sec = st.sidebar.selectbox("Online Security", ["No internet service", "No", "Yes"])
    backup = st.sidebar.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    dev_prot = st.sidebar.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech_sup = st.sidebar.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    tv = st.sidebar.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    movies = st.sidebar.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    # -- Akun --
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.sidebar.number_input("Biaya Bulanan ($)", min_value=0.0, value=50.0)
    total = st.sidebar.number_input("Total Biaya ($)", min_value=0.0, value=0.0)

    # Buat DataFrame
    data = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multi_lines,
        'InternetService': internet, 'OnlineSecurity': sec, 'OnlineBackup': backup,
        'DeviceProtection': dev_prot, 'TechSupport': tech_sup, 'StreamingTV': tv,
        'StreamingMovies': movies, 'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total
    }
    return pd.DataFrame(data, index=[0])

# Tampilkan Data Input
input_df = user_input_features()
st.subheader(" Data Pelanggan:")
st.write(input_df)

# 4. Prediksi
if st.button("🔍 Prediksi Sekarang"):
    prediction = model.predict(input_df)

    try:
        # Coba ambil probabilitas jika model mendukung
        proba = model.predict_proba(input_df)
        confidence = proba[0][prediction[0]] * 100
    except:
        confidence = 0

    st.subheader("💡 Hasil Analisis:")
    if prediction[0] == 1:
        st.error(f" CHURN DETECTED! Pelanggan ini berisiko tinggi berhenti berlangganan.")
    else:
        st.success(f" AMAN. Pelanggan ini diprediksi akan tetap setia.")

    if confidence > 0:
        st.info(f"Tingkat Keyakinan Model: {confidence:.1f}%")