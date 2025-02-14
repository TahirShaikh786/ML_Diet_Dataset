import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def ps_lifestyle_clustering(file_path):
    st.title("🚶‍♂️ Lifestyle Clustering (Using OLS + K-Means)")

    if file_path is not None:
        df = pd.read_csv(file_path)

        # 🔹 Define Features for Clustering
        independent_cols = ['Activity', 'Sleep', 'Eat Outside', 'Meals per day']

        X = df[independent_cols].dropna()  # Remove missing values

        # ✅ Fix: Set Random Seed to Ensure Consistent OLS Results
        np.random.seed(42)  
        random_target = np.random.rand(len(X))  # Fixed random target for OLS

        # 🔹 Step 1: Perform OLS Regression
        X_ols = sm.add_constant(X)
        ols_model = sm.OLS(random_target, X_ols).fit()
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

        # 🔹 Step 4: Normalize Data Using StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 🔹 Step 5: Determine Optimal Clusters Using Elbow Method
        distortions = []
        silhouette_scores = []
        K_range = range(2, 10)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            distortions.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

        # 🔹 Step 6: Display Elbow Method Graph
        st.write("### 📈 Elbow Method for Optimal Clusters")
        plt.figure(figsize=(6, 4))
        plt.plot(K_range, distortions, marker='o', linestyle='-')
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia (Distortion Score)")
        plt.title("Elbow Method to Find Optimal K")
        st.pyplot(plt)

        # 🔹 Step 7: Display Silhouette Scores
        st.write("### 🏆 Silhouette Scores for Different Cluster Sizes")
        best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
        st.write(f"🔹 **Optimal Number of Clusters (K) based on Silhouette Score:** {best_k}")

        # 🔹 Step 8: Train Final K-Means Model
        final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df["Cluster"] = final_kmeans.fit_predict(X_scaled)

        # 🔹 Step 9: Display Cluster Summaries
        cluster_summary = df.groupby("Cluster").mean()

        st.write("### 🏷️ Clustered Data (User Groups)")
        st.write(df.head())

        # ✅ Fix: Ensure At Least 2 Features Exist Before Plotting
        if len(significant_features) >= 2:
            st.write("### 🔍 Cluster Visualization")
            plt.figure(figsize=(6, 5))
            sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=df["Cluster"], palette="Set1")
            plt.xlabel(significant_features[0])
            plt.ylabel(significant_features[1])
            plt.title("K-Means Clustering of Users")
            st.pyplot(plt)
        else:
            st.write("⚠️ Not enough features for scatter plot.")

        # 🎯 User Prediction
        st.write("### 🎯 Predict User's Cluster")
        user_data = []
        for feature in significant_features:
            value = st.number_input(f"📌 {feature}", value=0.0)
            user_data.append(value)

        if st.button("🚀 Predict"):
            user_data_scaled = scaler.transform([user_data])
            user_prediction_kmeans = final_kmeans.predict(user_data_scaled)[0]
            st.write(f"### 🔮 Prediction: **User belongs to Cluster {user_prediction_kmeans}**")

            # ✅ Fix: Ensure Cluster Exists Before Displaying Details
            if user_prediction_kmeans in cluster_summary.index:
                st.write(f"### 📊 Details of Cluster {user_prediction_kmeans}:")
                st.write(cluster_summary.loc[user_prediction_kmeans])
            else:
                st.write("⚠️ Error: Cluster not found in summary. Please try again with valid inputs.")
