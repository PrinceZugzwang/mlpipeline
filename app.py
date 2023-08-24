import streamlit as st
import pandas as pd
import os
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Streamlit Sidebar
with st.sidebar:
    st.title("TradeSynth Analytics")
    choice = st.radio("Navigation", ["Upload", "ML"])

# Load Data
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

# Streamlit Logic
if choice == "Upload":
    st.title("Upload a dataset for modelling")
    file = st.file_uploader("Please upload your dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "ML":
    st.title("Applying Machine Learning!")
    target = st.selectbox("Select Your Target", df.columns)

    if target == "":
        st.warning("Please select a target column before proceeding.")
    else:
        X = df.drop(columns=[target])
        y = df[target]

        if y.dtypes == 'object':
            st.warning("Selected target column is categorical. Classification task will be performed.")
        else:
            st.warning("Selected target column is numerical. Regression task will be performed.")
            tpot = TPOTRegressor(generations=5,
                                 population_size=20,
                                 verbosity=2,
                                 random_state=1,
                                 config_dict='TPOT sparse')

            if y.dtypes == 'datetime64[ns]':
                y = (y - y.min()).dt.days  # Convert dates to days since the minimum date
            elif y.dtypes == 'timedelta64[ns]':
                y = y.dt.days  # Convert timedeltas to days

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            tpot.fit(X_train, y_train)

            y_pred = tpot.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.info(f"Mean Squared Error: {mse:.2f}")

            leaderboard = tpot.evaluated_individuals_
            st.info("Model Leaderboard:")
            st.dataframe(leaderboard)

            joblib.dump(tpot.fitted_pipeline_, 'best_model.pkl')
            with open('best_model.pkl', 'rb') as f:
                st.download_button("Download the Model", f, "best_model.pkl")
