import warnings
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

def ps_eating_out_preference(file_path):
    st.title("🍽️ Eating Preference Prediction (Using OLS + Decision Tree)")

    if file_path is not None:
        df = pd.read_csv(file_path)

        # 🔹 Define Target & Features
        dependent_col = "Eat Outside"
        independent_cols = ["Activity", "Sleep", "Drink Usage", "Medical Conditions", "Food Allergies"]

        X = df[independent_cols].dropna()  # Remove missing values
        y = df[dependent_col]

        # ✅ Fix: Set Random Seed to Ensure Consistent OLS Results
        np.random.seed(42)  

        # 🔹 Step 1: Perform OLS Regression to Select Significant Features
        X_ols = sm.add_constant(X)
        ols_model = sm.OLS(y, X_ols).fit()
        significant_results = ols_model.summary2().tables[1]

        # 🔹 Step 2: Increase p-value threshold to 0.2 (instead of 0.05)
        significant_results = significant_results[significant_results["P>|t|"] < 0.2]

        # 🔹 Step 3: Ensure Features Are Always Displayed
        st.write("### 📌 Significant Features from OLS Regression:")
        if not significant_results.empty:
            st.write(significant_results)
        else:
            st.write("⚠️ No significant features found (p < 0.2). Using all available features.")

        # Extract Feature Names
        significant_features = [f for f in significant_results.index if f in X.columns]

        # ✅ Fix: Ensure Consistent Features Are Used
        if not significant_features:
            significant_features = independent_cols  
            st.write("⚠️ No significant features found. Using all features.")

        X = df[significant_features]

        # 🔹 Step 4: Train-Test Split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 🔹 Step 5: Normalize Data Using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 🔹 Step 6: Train Decision Tree Classifier
        decision_tree_model = DecisionTreeClassifier(random_state=42)
        decision_tree_model.fit(X_train_scaled, y_train)

        # 🔹 Step 7: Model Predictions
        y_pred_dt = decision_tree_model.predict(X_test_scaled)

        # 🔹 Step 8: Evaluate Model Performance
        dt_accuracy = accuracy_score(y_test, y_pred_dt)

        st.write("### 📊 Model Performance")
        st.write(f"🔹 **Decision Tree Accuracy:** {dt_accuracy:.4f}")

        # 🔥 Confusion Matrix
        st.write("### 🔥 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_dt)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Mostly Outside", "Mostly Home"], 
                    yticklabels=["Mostly Outside", "Mostly Home"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)

        # 🔥 Classification Report
        st.write("### 📊 Classification Report")
        st.text(classification_report(y_test, y_pred_dt))

        # 🎯 User Prediction
        st.write("### 🎯 Predict Eating Preference")
        user_data = []
        for feature in significant_features:
            value = st.number_input(f"📌 {feature}", value=0.0)
            user_data.append(value)

        if st.button("🚀 Predict"):
            user_data_scaled = scaler.transform([user_data])
            user_prediction_dt = decision_tree_model.predict(user_data_scaled)[0]
            prediction_label = "🏠 Mostly Home" if user_prediction_dt == 1 else "🍽️ Mostly Outside"
            st.write(f"### 🔮 Prediction: **{prediction_label}**")
