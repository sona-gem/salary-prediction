import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Paylytics - Salary Predictor",
    page_icon="💼",
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #ffffff, #ffe0f0, #e0c3fc);
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }

    .app-title {
        font-size: 36px;
        font-weight: bold;
        color: #4a004e;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: #666;
        padding: 20px 0 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load all model-related files
@st.cache_resource
def load_models():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"❌ Could not load model files. Make sure all .pkl files are in the same folder!\n\nError: {e}")
        return None, None, None

def check_inputs(age, experience):
    if experience > (age - 16):
        return False, "Experience years seem too high for the given age!"
    return True, ""

def main():
    st.markdown('<h1 class="app-title">💼 Paylytics: Salary Predictor</h1>', unsafe_allow_html=True)
    st.write("Enter your details below to get a salary estimate!")

    # Load model, scaler, and feature columns
    model, scaler, feature_columns = load_models()
    if model is None:
        st.stop()

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("Your Details:")

    # User inputs
    age = st.slider("What's your age?", 18, 60, 25)
    experience = st.slider("How many years of work experience do you have?", 0, 30, 1)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    education = st.selectbox("Highest education level?", ['High School', "Bachelor's", "Master's", 'PhD', 'Not Specified'])

    # Dynamically extract job titles from feature columns
    job_prefix = 'Job Title_'
    job_columns = [col for col in feature_columns if col.startswith(job_prefix)]
    known_jobs = [col.replace(job_prefix, '') for col in job_columns]

    job_title = st.selectbox("What's your job title?", known_jobs,
                             index=known_jobs.index('Software Engineer') if 'Software Engineer' in known_jobs else 0)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮 Predict My Salary!", type="primary"):
        inputs_ok, error_message = check_inputs(age, experience)

        if not inputs_ok:
            st.error(error_message)
        else:
            with st.spinner("Calculating your salary..."):
                try:
                    input_df = pd.DataFrame([{
                        'Age': age,
                        'Years of Experience': experience,
                        'Gender': gender,
                        'Education Level': education,
                        'Job Title': job_title
                    }])

                    # Convert to dummy variables
                    input_dummies = pd.get_dummies(input_df)
                    input_dummies = input_dummies.reindex(columns=feature_columns, fill_value=0)

                    # Scale the input
                    input_scaled = scaler.transform(input_dummies)

                    # Predict salary
                    prediction = model.predict(input_scaled)[0]
                    
                    st.markdown(f"""
                        <div class="prediction-box" style="
                            background-color: #fff3f7;
                            border-left: 6px solid #d81b60;
                            padding: 20px;
                            border-radius: 12px;
                            margin-top: 25px;
                        ">
                            <h3 style="color:#ad1457;">💰 Estimated Annual Salary:</h3>
                            <p style="font-size: 28px; font-weight: bold; color: #6a1b9a;">${prediction:,.2f} / year</p>
                            <p style="font-size: 18px; margin-top: 10px; color: #4e148c;">≈ ${prediction/12:,.2f} per month</p>
                        </div>
                    """, unsafe_allow_html=True)

                    st.caption("⚠️ This is only an estimate based on historical trends and may vary in real scenarios.")

                except Exception as e:
                    st.error("Something went wrong during prediction.")
                    st.write(f"Error details: {e}")

    st.markdown('''
    <div class="footer">
        Made using Streamlit • Predictions are estimates based on historical data
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()