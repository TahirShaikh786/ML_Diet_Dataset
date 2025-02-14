import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def ps_calorie_prediction(file_path):
    st.title("🔥 Ideal Daily Calorie Prediction (Using OLS + Multiple Linear Regression)")

    if file_path is not None:
        df = pd.read_csv(file_path)

        # ✅ Add a Simulated 'Ideal Calories' Column
        np.random.seed(42)
        df["Ideal Calories"] = (
            10 * df["Weight"] +
            6.25 * df["Height"] -
            5 * df["Age"] +
            150 * df["Activity"] +
            100 * df["Sleep"] +
            np.random.normal(0, 100, df.shape[0])  # Adding some noise
        )

        # 🔹 Define Target & Features
        dependent_col = "Ideal Calories"
        independent_cols = ["Age", "Weight", "Height", "Activity", "Sleep"]

        X = df[independent_cols].dropna()  # Remove missing values
        y = df[dependent_col]

        # ✅ Fix: Set Random Seed to Ensure Consistent OLS Results
        np.random.seed(42)  

        # 🔹 Step 1: Perform OLS Regression to Select Significant Features
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

        # ✅ Fix: Ensure Consistent Features Are Used
        if not significant_features:
            significant_features = independent_cols  
            st.write("⚠️ No significant features found. Using all features.")

        X = df[significant_features]

        # 🔹 Step 4: Train-Test Split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 🔹 Step 5: Normalize Data Using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 🔹 Step 6: Train Multiple Linear Regression Model
        lin_reg = LinearRegression()
        lin_reg.fit(X_train_scaled, y_train)

        # 🔹 Step 7: Model Predictions
        y_pred_lin = lin_reg.predict(X_test_scaled)

        # 🔹 Step 8: Evaluate Model Performance
        lin_reg_r2 = r2_score(y_test, y_pred_lin)
        mae = mean_absolute_error(y_test, y_pred_lin)
        mse = mean_squared_error(y_test, y_pred_lin)
        rmse = np.sqrt(mse)

        st.write("### 📊 Model Performance")
        st.write(f"🔹 **R² Score:** {lin_reg_r2:.4f}")
        st.write(f"🔹 **Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"🔹 **Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"🔹 **Root Mean Squared Error (RMSE):** {rmse:.4f}")

        # 🎯 User Prediction
        st.write("### 🎯 Predict Your Ideal Daily Calorie Intake")
        user_data = []
        for feature in significant_features:
            value = st.number_input(f"📌 {feature}", value=0.0)
            user_data.append(value)

        if st.button("🚀 Predict"):
            user_data_scaled = scaler.transform([user_data])
            user_prediction_lin = lin_reg.predict(user_data_scaled)[0]
            st.write(f"### 🔮 Predicted Ideal Calorie Intake: **{user_prediction_lin:.2f} calories/day**")
