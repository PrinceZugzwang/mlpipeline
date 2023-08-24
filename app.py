import streamlit as st
import pandas as pd
import os
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    # Drop the target column before performing one-hot encoding
    X = df.drop(columns=[target])
    y = df[target]

    # Drop 'warnings' column before one-hot encoding
    if 'warnings' in X.columns:
      X = X.drop(columns=['warnings'])

    # Perform one-hot encoding for categorical variables
    X_encoded = pd.get_dummies(
      X, columns=X.select_dtypes(include='object').columns, drop_first=True)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=1)

    # Automated Model Selection with TPOT
    tpot = TPOTClassifier(generations=5,
                          population_size=20,
                          verbosity=2,
                          random_state=1,
                          config_dict='TPOT sparse')
    tpot.fit(X_train, y_train)

    # Model Evaluation
    y_pred = tpot.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.info(f"Accuracy: {accuracy:.2f}")

    # Show leaderboard with model performances
    leaderboard = pd.DataFrame({
        "Score": [entry["internal_cv_score"] for entry in tpot.evaluated_individuals_],
        "Model": [entry["pipeline"] for entry in tpot.evaluated_individuals_]
    })
    leaderboard = leaderboard.sort_values(by="Score", ascending=False)

    # Save and Download Best Model
    joblib.dump(tpot.fitted_pipeline_, 'best_model.pkl')
    with open('best_model.pkl', 'rb') as f:
      st.download_button("Download the Model", f, "best_model.pkl")



