import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def ps_medical_conditions(file_path):
    st.title("🏥 Medical Condition Prediction (Using OLS + Decision Tree)")

    if file_path is not None:
        df = pd.read_csv(file_path)

        # 🔹 Define Target & Features
        dependent_col = 'Medical Conditions'
        independent_cols = ['Weight', 'Height', 'Sleep', 'Drink Usage', 'Meals per day']

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

        # 🔹 Step 5: Train Decision Tree Classifier
        decision_tree = DecisionTreeClassifier(random_state=42)
        decision_tree.fit(X_train, y_train)

        # 🔹 Step 6: Model Predictions
        y_pred_dt = decision_tree.predict(X_test)

        # 🔹 Step 7: Evaluate Model Performance
        dt_accuracy = accuracy_score(y_test, y_pred_dt)

        st.write("### 📊 Model Performance")
        st.write(f"🔹 **Decision Tree Accuracy:** {dt_accuracy:.4f}")

        # 🔥 Confusion Matrix
        st.write("### 🔥 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_dt)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["No Medical Condition", "Has Medical Condition"], 
                    yticklabels=["No Medical Condition", "Has Medical Condition"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)

        # 🔥 Classification Report
        st.write("### 📊 Classification Report")
        st.text(classification_report(y_test, y_pred_dt))

        # 🔥 Display Decision Tree Structure
        st.write("### 🌳 Decision Tree Structure")
        tree_rules = export_text(decision_tree, feature_names=significant_features)
        st.text(tree_rules)

        # 🎯 User Prediction
        st.write("### 🎯 Predict Medical Condition")
        user_data = []
        for feature in significant_features:
            value = st.number_input(f"📌 {feature}", value=0.0)
            user_data.append(value)

        if st.button("🚀 Predict"):
            user_data_df = pd.DataFrame([user_data], columns=significant_features)
            user_prediction_dt = decision_tree.predict(user_data_df)[0]
            prediction_label = "✅ No Medical Condition" if user_prediction_dt == 0 else "⚠️ Has Medical Condition"
            st.write(f"### 🔮 Prediction: **{prediction_label}**")
