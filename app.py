# Import files 
import streamlit as st
import pandas as pd
import os 

# EDA import files  
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# ML imports 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR 

# ----------------- ML FUNCTIONS ----------------- #
def train_classification_models(X_train, X_test, y_train, y_test, selected_models):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }
    results = {}

    for model_name in selected_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[model_name] = {"Accuracy": acc, "Report": report}
    return results


def train_regression_models(X_train, X_test, y_train, y_test, selected_models):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(),
        "SVR": SVR()
    }
    results = {}

    for model_name in selected_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {"MSE": mse, "R2": r2}
    return results


def ml_pipeline(df, target_col, problem_type, selected_models):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run based on problem type
    if problem_type == "Classification":
        results = train_classification_models(X_train, X_test, y_train, y_test, selected_models)
    else:
        results = train_regression_models(X_train, X_test, y_train, y_test, selected_models)

    return results
# ------------------------------------------------ #

# ----------------- STREAMLIT APP ---------------- #
with st.sidebar:
    st.title("AutoMLpipeline")
    choice = st.radio("Navigation", ["Upload Your Dataset", "Exploratory Data Analysis", "ML", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit")


# Load dataset if exists
df = None
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)


if choice == "Upload Your Dataset":
    st.title("Upload Your Dataset Here")
    file = st.file_uploader("Upload your file (CSV/Excel)", type=["csv", "xlsx"])
    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df.head())


elif choice == "Exploratory Data Analysis":
    st.title("Automated Exploratory Data Analysis")
    if df is not None:
        st.subheader("Dataset Overview")
        st.write(df.shape)
        st.write(df.dtypes)
        st.write(df.describe())
        profile_report = ProfileReport(df, explorative=True)
        st_profile_report(profile_report) 
    else:
        st.warning("Please upload a dataset first.")


elif choice == "ML":
    st.title("Automated Machine Learning")

    if df is not None:
        target_col = st.selectbox("Select Target Column", df.columns)
        problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])

        if problem_type == "Classification":
            available_models = ["Logistic Regression", "Random Forest", "SVM"]
        else:
            available_models = ["Linear Regression", "Random Forest Regressor", "SVR"]

        selected_models = st.multiselect("Select Models to Train", available_models, default=available_models)

        if st.button("Train Models"):
            results = ml_pipeline(df, target_col, problem_type, selected_models)

            # Show results
            st.subheader("Model Performance")
            st.write(pd.DataFrame(results).T)

            best_model_choice = st.selectbox("Select Best Model", results.keys())
            st.success(f"You selected: {best_model_choice}")
    else:
        st.warning("Please upload a dataset first.") 


elif choice == "Download":
    st.title("Download Section")
    st.info("You can implement model or report download functionality here.")
