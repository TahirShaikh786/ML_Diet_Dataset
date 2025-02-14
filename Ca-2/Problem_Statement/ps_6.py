import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def ps_diet_plan_knn(file_path):
    st.title("🥗 Diet Plan Prediction (Using OLS + KNN)")

    if file_path is not None:
        df = pd.read_csv(file_path)

        # 🔹 Define Target & Features
        dependent_col = 'Diet Plan'
        independent_cols = ['Medical Conditions', 'Food Allergies', 'Age', 'Activity']

        X = df[independent_cols]
        y = df[dependent_col]

        # 🔹 Step 1: Perform OLS Regression
        X_ols = sm.add_constant(X)  
        ols_model = sm.OLS(y, X_ols).fit()
        significant_results = ols_model.summary2().tables[1]

        # 🔹 Step 2: Increase p-value threshold to 0.2 (instead of 0.05)
        significant_results = significant_results[significant_results['P>|t|'] < 0.2]

        # 🔹 Step 3: Ensure Features Are Always Displayed
        st.write("### 📌 Significant Features from OLS Regression:")
        if not significant_results.empty:
            st.write(significant_results)
        else:
            st.write("⚠️ No significant features found (p < 0.2). Using all available features.")

        # Extract Feature Names
        significant_features = [f for f in significant_results.index if f in X.columns]

        # If No Features Found, Use All Available Features
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

        # 🔹 Step 6: Train K-Nearest Neighbors (KNN) Classifier
        k = 5  # Number of neighbors
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train_scaled, y_train)

        # 🔹 Step 7: Model Predictions
        y_pred_knn = knn_model.predict(X_test_scaled)

        # 🔹 Step 8: Evaluate Model Performance
        knn_accuracy = accuracy_score(y_test, y_pred_knn)

        st.write("### 📊 Model Performance")
        st.write(f"🔹 **KNN Accuracy:** {knn_accuracy:.4f}")

        # 🔥 Confusion Matrix
        st.write("### 🔥 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_knn)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["No Diet Plan", "Follows Diet Plan"], 
                    yticklabels=["No Diet Plan", "Follows Diet Plan"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)

        # 🔥 Classification Report
        st.write("### 📊 Classification Report")
        st.text(classification_report(y_test, y_pred_knn))

        # 🎯 User Prediction
        st.write("### 🎯 Predict Diet Plan Adherence")
        user_data = []
        for feature in significant_features:
            value = st.number_input(f"📌 {feature}", value=0.0)
            user_data.append(value)

        if st.button("🚀 Predict"):
            user_data_scaled = scaler.transform([user_data])
            user_prediction_knn = knn_model.predict(user_data_scaled)[0]
            prediction_label = "✅ Follows a Diet Plan" if user_prediction_knn == 1 else "❌ Does Not Follow a Diet Plan"
            st.write(f"### 🔮 Prediction: **{prediction_label}**")
