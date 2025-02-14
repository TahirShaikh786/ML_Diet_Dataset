import streamlit as st
from Problem_Statement import ps_1, ps_2, ps_3, ps_4, ps_5, ps_6, ps_7, ps_8, ps_9, ps_10  # Import all problem statement files

file_path = "C:\\Users\\ms828\\Downloads\\ca2\\Ca-2\\Documents\\PreProcess_File.csv"

# 🔹 Dropdown for Selecting Problem Statement
def ProblemST():
    st.title("🔍 Select a Problem Statement")

    problem_statements = {
        "1. Predict Activity Level Using Random Forest & KNN": ps_1.ps,
        "2. Predict Diet Plan Adherence Using Logistic Regression & Random Forest": ps_2.ps_diet_plan,
        "3. predict an individual's ideal daily calorie intake Using Multiple Linear Regression": ps_3.ps_calorie_prediction,  # Replace with ps_3.ps
        "4. Classify Weight Goals (Lose/Gain) Using Knn": ps_4.ps_weight_goal,  # Replace with ps_4.ps
        "5. Cluster Users Based on Dietary Habits Using K-Means": ps_5.ps_lifestyle_clustering,  # Replace with ps_5.ps
        "6. Predict Whether a User Will Follow a Diet Plan Using Knn": ps_6.ps_diet_plan_knn,  # Replace with ps_6.ps
        "7. Predict whether a user has medical conditions Using Decision Tree": ps_7.ps_medical_conditions,  # Replace with ps_7.ps
        "8. Cluster users into different lifestyle groups Using K-Means Clustering": ps_8.ps_lifestyle_clustering,  # Replace with ps_8.ps
        "9. Recommend the best protein source for a user Using Knn": ps_9.ps_protein_recommendation,  # Replace with ps_9.ps
        "10. Predict whether an individual prefers eating meals at home or outside Using Descision Tree": ps_10.ps_eating_out_preference,  # Replace with ps_10.ps
    }

    selected_problem = st.selectbox("📌 Choose a problem statement:", list(problem_statements.keys()))

    if selected_problem:
        st.write(f"### 🚀 Running: {selected_problem}")
        model_function = problem_statements[selected_problem]
        
        if model_function is not None:
            model_function(file_path)  # Run the selected problem function
        else:
            st.write("⚠️ This problem statement is not yet implemented. Please check back later.")
