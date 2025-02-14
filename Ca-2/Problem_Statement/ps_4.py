import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def ps_weight_goal(file_path):
    st.title("⚖️ Weight Goal Prediction (Using OLS + KNN)")

    if file_path is not None:
        df = pd.read_csv(file_path)

        # 🔹 Define Target & Features
        dependent_col = 'lose/gain weight'
        independent_cols = ['Age', 'Weight', 'Height', 'Activity', 'Meals per day', 
                            'Diet Plan', 'Track your progress']

        X = df[independent_cols]
        y = df[dependent_col]

        # 🔹 Step 1: Remove Highly Correlated Features (Fixes Multicollinearity)
        correlation_matrix = X.corr()
        high_corr_features = [column for column in correlation_matrix.columns if any(correlation_matrix[column] > 0.85)]
        X = X.drop(columns=high_corr_features)  # Remove highly correlated features
        st.write("### ❌ Removed Highly Correlated Features:", high_corr_features)

        # 🔹 Step 2: Perform OLS Regression
        X_ols = sm.add_constant(X)  # Add constant for OLS regression
        ols_model = sm.OLS(y, X_ols).fit()
        significant_results = ols_model.summary2().tables[1]

        # 🔹 Step 3: Increase p-value threshold to 0.2 (instead of 0.05)
        significant_results = significant_results[significant_results['P>|t|'] < 0.2]

        # 🔹 Step 4: Ensure Features Are Always Displayed
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

        # 🔹 Train-Test Split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 🔹 Step 5: Normalize Data Using MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 🔹 Step 6: Train K-Nearest Neighbors (KNN) Classifier
        knn_model = KNeighborsClassifier(n_neighbors=5)
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Maintain", "Gain", "Lose"], 
                    yticklabels=["Maintain", "Gain", "Lose"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)

        # 🔥 Classification Report
        st.write("### 📊 Classification Report")
        st.text(classification_report(y_test, y_pred_knn))

        # 🎯 User Prediction
        st.write("### 🎯 Predict Weight Goal")
        user_data = []
        for feature in significant_features:
            value = st.number_input(f"📌 {feature}", value=0.0)
            user_data.append(value)

        if st.button("🚀 Predict"):
            user_data_scaled = scaler.transform([user_data])
            user_prediction_knn = knn_model.predict(user_data_scaled)[0]
            prediction_label = "⚖️ Maintain Weight" if user_prediction_knn == 0 else "📈 Gain Weight" if user_prediction_knn == 1 else "📉 Lose Weight"
            st.write(f"### 🔮 Prediction: **{prediction_label}**")
