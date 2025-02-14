import streamlit as st
import pandas as pd

# Load the dataset
st.title("Data Cleaning & Transformation")
uploaded_file = "C:\\Users\\ms828\\Downloads\\ca2\\Ca-2\\Documents\\Diet_Plan_Responses.csv"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data")
    st.dataframe(df.head())

    # Define columns to transform
    column_to_transform = [
        'Medical Conditions',
        'Food Allergies',
        'Diet Plan',
        'Focuses short-long-term health',
        'Track your progress',
        'lose/gain weight'
    ]
    
    st.write("### Category Column Data Before Transformation")
    st.dataframe(df[column_to_transform].head())
    
    for col in column_to_transform:
        if col in df.columns:
            df[col] = df[col].replace({'Yes': 1, 'No': 0})

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
    
    st.write("### Category Column Data After Transformation")
    st.dataframe(df[column_to_transform].head())
    
    st.write("### Transformed Data")
    st.dataframe(df.head())
    
    # Save transformed data
    output_path = "C:\\Users\\ms828\\Downloads\\ca2\\Ca-2\\Documents\\PreProcess_File.csv"
    df.to_csv(output_path, index=False)
    st.download_button("Download Transformed Data", data=df.to_csv(index=False), file_name=output_path, mime="text/csv")
    
    st.success(f"The dataset with numeric data has been saved to {output_path}.")
  
def get_outputpath():
    return "C:\\Users\\ms828\\Downloads\\ca2\\Ca-2\\Documents\\PreProcess_File.csv"
