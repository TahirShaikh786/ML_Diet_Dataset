import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def ps_diet_plan(file_path):
    st.title("🥗 Diet Plan Prediction")

    if file_path is not None:
        df = pd.read_csv(file_path)

        # 🔹 Define Target & Features
        dependent_col = 'Diet Plan'
        independent_cols = ['Age', 'Weight', 'Height', 'Sleep', 'Drink Usage', 'Eat Outside', 
                            'Meals per day', 'Source of protein', 'Activity', 'lose/gain weight']

        X = df[independent_cols]
        X = sm.add_constant(X)  # Add constant for OLS regression
        y = df[dependent_col]

        # 🔹 Feature Selection using OLS Regression
        regressor_OLS = sm.OLS(endog=y, exog=X).fit()
        significant_results = regressor_OLS.summary2().tables[1]
        significant_results = significant_results[significant_results['P>|t|'] < 0.5]  # Adjusted threshold

        st.write("### 📌 Significant Features from OLS Regression:")
        st.write(significant_results)

        # 🔹 Use Significant Features or Default to All Features
        significant_features = [f for f in significant_results.index if f in df.columns]
        if not significant_features:
            significant_features = independent_cols
            st.write("⚠️ No significant features found. Using all available features.")

        X = df[significant_features]

        # 🔹 Train-Test Split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 🔹 Standardizing Data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 🔹 Model 1: Logistic Regression
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        y_pred_log = log_reg.predict(X_test)
        log_reg_accuracy = accuracy_score(y_test, y_pred_log)

        # 🔹 Model 2: Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)

        # 🔹 Model Accuracy
        st.write("### 📊 Model Accuracy")
        st.write(f"🔹 **Logistic Regression Accuracy:** {log_reg_accuracy:.4f}")
        st.write(f"🔹 **Random Forest Accuracy:** {rf_accuracy:.4f}")

        # 🔥 Confusion Matrices
        st.write("### 🔥 Confusion Matrix - Logistic Regression")
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)

        st.write("### 🔥 Confusion Matrix - Random Forest")
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)

        st.write("### ℹ️ Input Instructions")
    
        st.write("**🟢 Activity Level (Choose the corresponding number):**")
        st.write("""
        1 → 🏠 Sedentary (Little or no exercise)  
        2 → 🚶 Lightly Active (Exercise 1-3 days/week)  
        3 → 🏋️ Moderately Active (Exercise 3-5 days/week)  
        4 → 🔥 Very Active (Exercise 6-7 days/week)  
        """)

        st.write("**🟢 Source of Protein (Choose the corresponding number):**")
        st.write("""
        1 → Dairy (Cheese, Yogurt)  
        2 → Protein Supplements (Whey, Plant-based protein)  
        3 → Non-Veg  
        4 → Plant-based (Legumes, Tofu, Tempeh)  
        """)

        # 🎯 User Prediction
        st.write("### 🎯 Predict Diet Plan Adherence")
        user_data = []
        for feature in X.columns:
            value = st.number_input(f"📌 {feature}", value=0.0)
            user_data.append(value)

        if st.button("🚀 Predict with Logistic Regression"):
            user_data_scaled = scaler.transform([user_data])
            user_prediction_log = log_reg.predict(user_data_scaled)[0]
            prediction_label = "✅ Follows a Diet Plan" if user_prediction_log == 1 else "❌ Does Not Follow a Diet Plan"
            st.write(f"### 🔮 Prediction (Logistic Regression): **{prediction_label}**")

        if st.button("🚀 Predict with Random Forest"):
            user_data_scaled = scaler.transform([user_data])
            user_prediction_rf = rf.predict(user_data_scaled)[0]
            prediction_label = "✅ Follows a Diet Plan" if user_prediction_rf == 1 else "❌ Does Not Follow a Diet Plan"
            st.write(f"### 🔮 Prediction (Random Forest): **{prediction_label}**")
