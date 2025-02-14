import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 🔹 Mapping for Activity Levels
activity_labels = {
    1: "🏠 Sedentary (Little or no exercise)",
    2: "🚶 Lightly Active (Exercise 1-3 days/week)",
    3: "🏋️ Moderately Active (Exercise 3-5 days/week)",
    4: "🔥 Very Active (Exercise 6-7 days/week)"
}

def ps(file_path):
    st.title("🏃 Activity Level Prediction")

    if file_path is not None:
        df = pd.read_csv(file_path)

        # 🔹 Check available columns
        st.write("📌 Available Columns in Dataset:")
        st.write(df.columns.tolist())

        # 🔹 Define Target & Features
        dependent_col = 'Activity'
        available_columns = df.columns.tolist()
        
        # Ensure columns exist before selecting them
        independent_cols = [col for col in ['Age', 'Weight', 'Height', 'Sleep', 'Drink Usage', 'Eat Outside', 'Meals per day', 
                            'Source of protein', 'lose/gain weight', 'Diet Plan'] if col in available_columns]

        # Select only available features
        X = df[independent_cols]
        X = sm.add_constant(X)  # Add constant for OLS regression
        y = df[dependent_col]

        # 🔹 Feature Selection using OLS Regression
        regressor_OLS = sm.OLS(endog=y, exog=X).fit()
        significant_results = regressor_OLS.summary2().tables[1]
        significant_results = significant_results[significant_results['P>|t|'] < 0.5]  # Adjusted threshold for more features

        st.write("### 📌 Significant Features from OLS Regression:")
        st.write(significant_results)

        # 🔹 Use Significant Features or Default to All Available Features
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

        # 🔹 Model 1: Random Forest Classifier (Optimized)
        rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)

        # 🔹 Model 2: K-Nearest Neighbors (KNN) - Hyperparameter Optimization
        param_grid = {'n_neighbors': range(1, 20)}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_k = grid_search.best_params_['n_neighbors']

        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        knn_accuracy = accuracy_score(y_test, y_pred_knn)

        # 🔹 Model Accuracy
        st.write("### 📊 Model Accuracy")
        st.write(f"🔹 **Random Forest Accuracy:** {rf_accuracy:.4f}")
        st.write(f"🔹 **KNN Accuracy (Optimized `k={best_k}`):** {knn_accuracy:.4f}")

        # 🔥 Confusion Matrix for Random Forest
        st.write("### 🔥 Confusion Matrix - Random Forest")
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)

        # 🔍 Feature Importance from Random Forest
        st.write("### 🔍 Feature Importance (Random Forest)")
        feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=feature_importance, y=feature_importance.index, palette="coolwarm")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.title("Feature Importance in Predicting Activity Level")
        st.pyplot(plt)

        # 🎯 User Prediction
        st.write("### 🎯 Predict Activity Level")
        user_data = []
        for feature in X.columns:
            value = st.number_input(f"📌 {feature}", value=0.0)
            user_data.append(value)

        if st.button("🚀 Predict with Random Forest"):
            user_data_scaled = scaler.transform([user_data])
            user_prediction_rf = rf.predict(user_data_scaled)[0]
            activity_level_rf = activity_labels.get(user_prediction_rf, "Unknown Activity Level")
            st.write(f"### 🔮 Prediction (Random Forest): **{activity_level_rf}**")

        if st.button("🚀 Predict with KNN"):
            user_data_scaled = scaler.transform([user_data])
            user_prediction_knn = knn.predict(user_data_scaled)[0]
            activity_level_knn = activity_labels.get(user_prediction_knn, "Unknown Activity Level")
            st.write(f"### 🔮 Prediction (KNN): **{activity_level_knn}**")
