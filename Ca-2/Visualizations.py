import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder


def visualize_data(file_path):
    st.title("🔍 Data Visualization & Insights")

    # Load the dataset
    df = pd.read_csv(file_path)

    # Define Quantitative and Qualitative data
    quantitative_columns = ['Age', 'Weight', 'Height']
    qualitative_columns = ['Gender', 'Activity', 'Sleep', 'Medical Conditions', 'Food Allergies',
                           'Source of protein', 'lose/gain weight', 'Diet Plan',
                           'Focuses short-long-term health', 'Track your progress',
                           'Eat Outside', 'Drink Usage', 'Meals per day']

    st.write("This section presents visual insights from the dataset, including distributions, trends, and correlations.")

    # ---- Pie Charts for Categorical Data ----
    st.header("📊 Pie Charts for Categorical Data")
    pie_columns = ['Activity', 'Sleep', 'Eat Outside', 'Drink Usage', 'Meals per day']

    for col in pie_columns:
        counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=[f'{label} ({count})' for label, count in zip(counts.index, counts)],
               autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2', n_colors=len(counts)))
        ax.set_title(f"{col} Distribution")
        st.pyplot(fig)

    # ---- Count Plots for Categorical Data ----
    st.header("📌 Categorical Data Distributions (Bar Charts)")
    for col in qualitative_columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x=col, palette="Set2", order=df[col].value_counts().index)
        ax.set_title(f"Distribution of {col}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

    # ---- Histograms for Quantitative Data ----
    st.header("📈 Histograms for Numerical Data")
    hist_columns = ['Age', 'Weight', 'Height']
    hist_colors = ['skyblue', 'lightgreen', 'lightcoral']

    for col, color in zip(hist_columns, hist_colors):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df[col], bins=10, color=color, edgecolor='black', alpha=0.7)
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df[col].mean():.2f}')
        ax.legend()
        st.pyplot(fig)

    # ---- Box Plots for Numerical Data ----
    st.header("📦 Box Plots for Outlier Detection")
    for col in quantitative_columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(y=df[col], palette="Set2", ax=ax)
        ax.set_title(f"Box Plot for {col}")
        st.pyplot(fig)

    # ---- Violin Plots for Better Distribution Analysis ----
    st.header("🎻 Violin Plots for Numerical Data")
    for col in quantitative_columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(y=df[col], palette="Set1", ax=ax)
        ax.set_title(f"Violin Plot for {col}")
        st.pyplot(fig)

    # ---- Stacked Bar Chart for Categorical Relationships ----
    st.header("📊 Stacked Bar Chart (Diet Plan vs. Activity)")
    diet_activity = pd.crosstab(df['Diet Plan'], df['Activity'])
    diet_activity.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title("Diet Plan by Activity Level")
    plt.xlabel("Diet Plan")
    plt.ylabel("Count")
    st.pyplot(plt)

    # ---- Pair Plot for Relationships in Quantitative Data ----
    st.header("🔗 Pair Plot for Relationships in Numerical Data")
    fig = sns.pairplot(df[quantitative_columns], diag_kind='kde', corner=True)
    st.pyplot(fig)

    # ---- Tree Map for Source of Protein ----
    st.header("🌳 Tree Map for Source of Protein")
    counts = df['Source of protein'].value_counts().reset_index()
    counts.columns = ['Source of protein', 'Count']
    fig = px.treemap(counts,
                     path=['Source of protein'],
                     values='Count',
                     color='Count',
                     color_continuous_scale='Blues',
                     title='Tree Map for Source of Protein')
    st.plotly_chart(fig)

    # ---- Heatmap for Categorical Data ----
    st.header("🔥 Correlation Heatmap for Categorical Data")
    label_encoder = LabelEncoder()
    encoded_qualitative_data = df[qualitative_columns].apply(label_encoder.fit_transform)
    categorical_correlation_matrix = encoded_qualitative_data.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(categorical_correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap for Categorical Data")
    st.pyplot(fig)
