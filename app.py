import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="✈️",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("Customertravel.csv")

model = load_model()
df = load_data()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("✈️ Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Explorer", "🤖 Predict Churn"])

# ── HOME PAGE ─────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("✈️ Customer Churn Prediction")
    st.subheader("Using Random Forest | End-to-End ML & Deployment")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📁 Total Records", "954")
    col2.metric("🎯 Model Accuracy", "78.89%")
    col3.metric("📈 AUC Score", "0.8542")
    col4.metric("⚡ F1-Score (Churn)", "0.80")

    st.markdown("---")
    st.markdown("""
    ### What is Customer Churn?
    Churn means customers who **stop using a service**.
    - Costs **5–7×** more to acquire than retain a customer
    - Even **5% better retention** → up to **95% profit increase**
    - Early prediction enables **proactive engagement**

    ### About This App
    This app uses a **Random Forest** classifier trained on travel customer data
    to predict whether a customer will churn. Use the sidebar to:
    - 📊 **Explore** the dataset and visualizations
    - 🤖 **Predict** churn for a new customer
    """)

# ── DATA EXPLORER PAGE ────────────────────────────────────────────────────────
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("---")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        counts = df['Target'].value_counts()
        ax.pie(counts, labels=['Not Churned', 'Churned'],
               autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        ax.set_title("Churn Rate")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Age Distribution by Churn")
        fig, ax = plt.subplots()
        df[df['Target'] == 0]['Age'].hist(alpha=0.6, label='Not Churned',
                                           color='#4CAF50', ax=ax, bins=20)
        df[df['Target'] == 1]['Age'].hist(alpha=0.6, label='Churned',
                                           color='#F44336', ax=ax, bins=20)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Churn by Frequent Flyer")
        fig, ax = plt.subplots()
        ff_churn = df.groupby('FrequentFlyer')['Target'].mean().reset_index()
        sns.barplot(data=ff_churn, x='FrequentFlyer', y='Target',
                    palette=['#4CAF50', '#F44336', '#FF9800'], ax=ax)
        ax.set_ylabel("Churn Rate")
        ax.set_title("Frequent Flyer vs Churn Rate")
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Churn by Income Class")
        fig, ax = plt.subplots()
        inc_churn = df.groupby('AnnualIncomeClass')['Target'].mean().reset_index()
        sns.barplot(data=inc_churn, x='AnnualIncomeClass', y='Target',
                    palette='coolwarm', ax=ax)
        ax.set_ylabel("Churn Rate")
        ax.set_title("Income Class vs Churn Rate")
        plt.xticks(rotation=15)
        st.pyplot(fig)
        plt.close()

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# ── PREDICT PAGE ──────────────────────────────────────────────────────────────
elif page == "🤖 Predict Churn":
    st.title("🤖 Predict Customer Churn")
    st.markdown("Fill in the customer details below and click **Predict**.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=85, value=35)
        frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes", "No Record"])
        annual_income = st.selectbox("Annual Income Class",
                                      ["Low Income", "Middle Income", "High Income"])

    with col2:
        services_opted = st.slider("Services Opted", min_value=1, max_value=9, value=3)
        social_media = st.selectbox("Account Synced to Social Media", ["No", "Yes"])
        hotel_booked = st.selectbox("Booked Hotel or Not", ["No", "Yes"])

    # Encode inputs to match training data encoding
    def encode_input(age, frequent_flyer, annual_income,
                     services_opted, social_media, hotel_booked):
        ff_map = {"No": 0, "Yes": 1, "No Record": 2}
        inc_map = {"Low Income": 0, "Middle Income": 1, "High Income": 2}
        yn_map = {"No": 0, "Yes": 1}

        return pd.DataFrame([[
            age,
            ff_map[frequent_flyer],
            inc_map[annual_income],
            services_opted,
            yn_map[social_media],
            yn_map[hotel_booked]
        ]], columns=[
            'Age', 'FrequentFlyer', 'AnnualIncomeClass',
            'ServicesOpted', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot'
        ])

    if st.button("🔍 Predict", use_container_width=True):
        input_df = encode_input(age, frequent_flyer, annual_income,
                                services_opted, social_media, hotel_booked)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ **This customer is likely to CHURN**")
            st.metric("Churn Probability", f"{probability[1]*100:.1f}%")
        else:
            st.success(f"✅ **This customer is likely to STAY**")
            st.metric("Retention Probability", f"{probability[0]*100:.1f}%")

        # Probability bar
        col_a, col_b = st.columns(2)
        col_a.metric("P(Not Churned)", f"{probability[0]*100:.1f}%")
        col_b.metric("P(Churned)", f"{probability[1]*100:.1f}%")

        # Feature importance
        st.markdown("---")
        st.subheader("📌 Feature Importance")
        feature_names = ['Age', 'FrequentFlyer', 'AnnualIncomeClass',
                         'ServicesOpted', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot']
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(fi_df['Feature'], fi_df['Importance'], color='#1976D2')
        ax.set_xlabel("Importance Score")
        ax.set_title("What Drives Churn?")
        st.pyplot(fig)
        plt.close()
