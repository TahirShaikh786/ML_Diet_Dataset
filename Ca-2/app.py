import streamlit as st
import pandas as pd
from Preprocess import load_and_preprocess_data
from Transformation import get_outputpath
from Visualizations import visualize_data
from ProblemST import ProblemST


# File path for the CSV
file_path = "C:\\Users\\ms828\\Downloads\\ca2\\Ca-2\\Documents\\Diet_Plan_Responses.csv"

# Homepage content
def home_page():
    st.title("Diet & Lifestyle Insights: Understanding Personal Nutrition Choices")
    st.subheader("Welcome to the Diet & Lifestyle Insights App!")
    st.write("""
        This platform explores how individuals make food-related choices and the impact of lifestyle habits on health.
        We gathered data from a diverse group of users to understand how they approach their daily nutrition and wellness routines.
    """)
    
    st.write("### Key Insights from the Data:")
    
    st.write("""
        - **Age**: The age of the respondent, which helps us understand the relationship between age and various lifestyle choices.
        - **Gender**: The gender of the respondent, as this may influence food consumption patterns and lifestyle habits.
        - **Weight**: The weight of the individual (in kg), which plays a role in understanding their dietary habits and health goals.
        - **Height**: The height of the individual (in cm), which, together with weight, helps in determining Body Mass Index (BMI).
        - **Activity**: The level of physical activity of the individual, ranging from sedentary to very active. This column helps us understand how exercise correlates with other lifestyle habits like diet and sleep.
        - **Sleep**: The number of hours the individual sleeps, categorized as "Less than 5 hours", "7-8 hours", etc. Sleep is a crucial factor in determining overall health and nutrition choices.
        - **Drink Usage**: The number of liters of water the individual consumes per day, indicating hydration habits.
        - **Eat Outside**: A rating (from 1 to 5) indicating how often the individual eats outside, giving insights into their food preferences and social habits.
        - **Medical Conditions**: Indicates whether the respondent has any medical conditions that may affect their dietary choices (e.g., diabetes, heart disease, etc.).
        - **Food Allergies**: Whether the respondent has any food allergies that impact their food selection and nutrition choices.
        - **Meals per Day**: The number of meals the individual consumes each day, providing insight into eating frequency and habits.
        - **Source of Protein**: The primary source of protein in the individual’s diet (e.g., Dairy, Plant-based, or Protein Supplements). This can show preferences in dietary choices based on lifestyle or dietary restrictions.
        - **Lose/Gain Weight**: Whether the individual is trying to lose or gain weight. This column provides insight into their health goals and motivation behind their diet and exercise choices.
        - **Diet Plan**: Whether the individual follows a structured diet plan (e.g., keto, intermittent fasting, balanced diet). This can be linked to their health goals or personal preferences.
        - **Focuses Short-Long-Term Health**: Indicates whether the individual focuses more on short-term or long-term health goals. This can help identify whether they prioritize quick results or sustainable health outcomes.
        - **Track Your Progress**: Whether the individual actively tracks their diet and health progress (e.g., weight tracking, food logging, etc.).
    """)

    st.write("""
        ### Overview of the Data:
        
        This dataset provides a detailed view of the eating habits and lifestyle choices of individuals. We collected information on their age, weight, height, physical activity, sleep patterns, and more. This helps us identify potential correlations between various factors such as:

        - **Physical activity** and the frequency of eating outside, meals per day, and hydration habits.
        - **Sleep** patterns and their relationship with meal frequency, hydration, and protein sources.
        - **Health goals** such as losing or gaining weight, and how these relate to their diet plan, protein sources, and whether they track their progress.
        - **Dietary preferences** such as protein sources, meal frequency, and eating outside, which can be linked to overall lifestyle choices and long-term health focus.

        By analyzing this data, we aim to better understand how different factors like activity level, sleep, and health goals shape the eating habits and lifestyle choices of individuals.

    """)

def transformation_page():
    output_path = get_outputpath()
    raw_file = pd.read_csv(file_path)
    processed_file = pd.read_csv(output_path)

    st.write("### Raw Data Before Transformation")
    st.dataframe(raw_file.head())

    st.write("### Processed Data After Transformation")
    st.dataframe(processed_file.head())

    # Display Category Mappings
    st.write("### Category Mappings (Before and After Transformation)")

    category_mappings = {
        "Medical Conditions": {"Yes": 1, "No": 0},
        "Food Allergies": {"Yes": 1, "No": 0},
        "Diet Plan": {"Yes": 1, "No": 0},
        "Focuses short-long-term health": {"Yes": 1, "No": 0},
        "Track your progress": {"Yes": 1, "No": 0},
        "Lose/Gain Weight": {"Yes": 1, "No": 0},
        "Gender": {"Male": 1, "Female": 2},
        "Activity": {
            "Sedentary (little or no exercise)": 1,
            "Lightly active (light exercise 1-3 days/week)": 2,
            "Moderately active (moderate exercise 3-5 days/week)": 3,
            "Very active (intense exercise 6-7 days/week)": 4,
        },
        "Sleep": {
            "Less than 5 hours": 4,
            "5-6 hours": 5,
            "7-8 hours": 7,
            "Over 8 hours": 8,
        },
        "Source of Protein": {
            "Dairy (Cheese, Yogurt)": 1,
            "Protein supplements (Whey, Plant-based protein)": 2,
            "Non-Veg": 3,
            "Plant-based (Legumes, Tofu, Tempeh)": 4,
        },
    }

    # Display mappings
    for col, mapping in category_mappings.items():
        st.write(f"**{col} Mapping:**")
        st.write(mapping)

# Preprocess Data Page
def preprocess_page():
    st.write(load_and_preprocess_data(file_path))

# Sidebar Navigation
def main():
    st.sidebar.title("Navigation")
    
    # Home Page Link
    page = st.sidebar.radio("Go to", ("Home", "Data Transformation", "Preprocess Data", "Data Visualization", "Problem Statement"))

    # Load data
    df = pd.read_csv(file_path)

    if page == "Home":
        home_page()
    elif page == "Data Transformation":
        transformation_page()
    elif page == "Preprocess Data":
        preprocess_page()
    elif page == "Data Visualization":
        visualize_data(file_path)
        print("Hello")
    elif page == "Problem Statement":
        ProblemST()

if __name__ == "__main__":
    main()
