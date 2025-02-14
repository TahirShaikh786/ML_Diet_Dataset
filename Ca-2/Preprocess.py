import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def load_and_preprocess_data(file_path):
    st.title("📊 Diet Plan Data Preprocessing & Transformation")

    if file_path is not None:
        df = pd.read_csv(file_path)

        # 🔹 Step 1: Display Raw Data
        st.write("### 📂 Raw Dataset Preview")
        st.dataframe(df.head())

        # Select only numeric columns for analysis
        numeric_df = df.select_dtypes(include=[np.number])

        # 🔹 Step 2: Perform Data Analysis (Before Transformation)
        st.write("## 📊 Exploratory Data Analysis (EDA)")

        # 🔹 Basic Statistics
        st.write("### 🔢 Basic Statistical Metrics")

        # Calculate metrics
        mean_values = numeric_df.mean()
        median_values = numeric_df.median()
        mode_values = numeric_df.mode().iloc[0]
        std_dev = numeric_df.std()
        variance = numeric_df.var()

        # Combine all statistics into a single table
        stats_df = pd.DataFrame({
            "Mean": mean_values,
            "Median": median_values,
            "Mode": mode_values,
            "Standard Deviation": std_dev,
            "Variance": variance
        })

        st.write(stats_df)

        # 🔹 Skewness & Kurtosis
        st.write("### 📈 Skewness & Kurtosis")
        skewness = numeric_df.apply(lambda x: skew(x.dropna()))
        kurtosis_values = numeric_df.apply(lambda x: kurtosis(x.dropna()))

        st.write("🔹 **Skewness**")
        st.write(skewness)

        st.write("🔹 **Kurtosis**")
        st.write(kurtosis_values)

        # 🔹 Duplicates Check
        duplicates = numeric_df.duplicated().sum()
        st.write(f"### 🔍 Number of Duplicates: {duplicates}")

        # 🔹 Correlation Matrix
        st.write("### 🔗 Correlation Matrix")
        correlation_matrix = numeric_df.corr()
        st.write(correlation_matrix)

        # 🔹 Key Correlations
        st.write("### 🔥 Top 10 High-Correlation Pairs")
        high_correlation_pairs = correlation_matrix.unstack().sort_values(ascending=False)
        high_correlation_pairs = high_correlation_pairs[high_correlation_pairs < 1]  # Remove self-correlations
        st.write(high_correlation_pairs.head(10))

        # 🔹 Step 3: Identify Trends & Patterns
        st.write("## 📊 Trends and Patterns in the Dataset")

        # Activity Level vs. Meals per Day
        st.write("🔹 **Activity Level and Meals per Day**")
        activity_meals = df.groupby("Activity")["Meals per day"].mean()
        st.write(activity_meals)

        # Sleep and Hydration Trends
        st.write("🔹 **Sleep Duration and Water Intake**")
        sleep_hydration = df.groupby("Sleep")["Drink Usage"].mean()
        st.write(sleep_hydration)

        # Weight vs. Preferred Protein Source
        st.write("🔹 **Weight vs. Preferred Protein Source**")
        weight_protein = df.groupby("Source of protein")["Weight"].mean()
        st.write(weight_protein)

        # Eating Out Frequency and Diet Plan
        st.write("🔹 **Eating Out Frequency vs. Diet Plan Preference**")
        eat_out_diet = df.groupby("Eat Outside")["Diet Plan"].value_counts(normalize=True)
        st.write(eat_out_diet)

        # Weight Goals Trends
        st.write("🔹 **Weight Goal Trends (Lose/Gain)**")
        weight_goals = df["lose/gain weight"].value_counts()
        st.write(weight_goals)

        # 🔹 Quartiles and Deciles
        st.write("### 🔢 Quartiles & Deciles")
        st.write("📌 **Quartiles**")
        st.write(numeric_df.quantile([0.25, 0.5, 0.75]))

        st.write("📌 **Deciles**")
        st.write(numeric_df.quantile(np.arange(0.1, 1, 0.1)))

        # 🔹 Step 4: Perform Data Transformation (Encoding Categorical Variables)
        st.write("## 🔄 Data Transformation (Encoding Categorical Variables)")

        column_to_transform = [
            'Medical Conditions',
            'Food Allergies',
            'Diet Plan',
            'Focuses short-long-term health',
            'Track your progress',
            'lose/gain weight'
        ]

        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 2})

        if 'Activity' in df.columns:
            df['Activity'] = df['Activity'].replace({
                'Sedentary (little or no exercise)': 1,
                'Lightly active (light exercise 1-3 days/week)': 2,
                'Moderately active (moderate exercise 3-5 days/week)': 3,
                'Very active (intense exercise 6-7 days/week)': 4
            })

        if 'Sleep' in df.columns:
            df['Sleep'] = df['Sleep'].replace({
                'Less than 5 hours': 4,
                '5-6 hours': 5,
                '7-8 hours': 7,
                'Over 8 hours': 8
            })

        if 'Source of protein' in df.columns:
            df['Source of protein'] = df['Source of protein'].replace({
                'Dairy (Cheese, Yogurt)': 1,
                'Protein supplements (Whey, Plant-based protein)': 2,
                'Non-Veg': 3,
                'Plant-based (Legumes, Tofu, Tempeh)': 4
            })

        for col in column_to_transform:
            if col in df.columns:
                df[col] = df[col].replace({'Yes': 1, 'No': 0})

        # 🔹 Step 5: Display Transformed Data
        st.write("### 🔄 Transformed Data Preview")
        st.dataframe(df.head())

        # Save transformed data
        output_path = "C:\\Users\\ms828\\Downloads\\ca2\\Ca-2\\Documents\\PreProcess_File.csv"
        df.to_csv(output_path, index=False)

        # Download Button
        st.download_button(
            "⬇ Download Transformed Data",
            data=df.to_csv(index=False),
            file_name="PreProcess_File.csv",
            mime="text/csv"
        )

        st.success(f"✅ The transformed dataset has been saved to `{output_path}`")

# Function to return output file path
def get_outputpath():
    return "C:\\Users\\ms828\\Downloads\\ca2\\Ca-2\\Documents\\PreProcess_File.csv"
