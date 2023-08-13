import streamlit as st
import pandas as pd
import os
import h2o
from h2o.automl import H2OAutoML
import matplotlib
# import sweetviz as sv

with st.sidebar:
    st.title("TradeSynth Analytics")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download", "Predictions"])
    st.info("This app allows you to build an Automated ML Pipeline!")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col = None)
if choice == "Upload":
    st.title("Upload a dataset for modelling")
    file = st.file_uploader("Please upload your dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
         

# if choice == "Profiling":
    # st.title("Explore the Data with Automated Analysis!")
    # profile_report = sv.analyze(df, title="Profiling Report")
    # profile_report.show_html()


if choice == "ML":
    st.title("Applying Machine Learning!")
    target = st.selectbox("Select Your Target", df.columns)
    h2o.init()
    if st.button("Train Model"):
        df = h2o.H2OFrame(df)
        st.info("done1")
        aml = H2OAutoML(max_models =25,
                    balance_classes=True,
                    seed = 1)
        st.info("done2")
        # modeling and training
        aml.train(training_frame = df, y = target)
        st.info("done3")
        lb = aml.leaderboard
        st.info("This is how different models performed")
        lb.head(rows=lb.nrows)
        # generate best model
        best_model = aml.get_best_model()
        st.info("This is your best model for the situation")
        print(best_model)
        h2o.save_model(best_model, "best_model")

if choice == "Download":
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download the Model", f, "best_model.pkl")

if choice == "Predictions":
    st.title("Upload a dataset for making predictions on")
    file = st.file_uploader("Please upload your dataset")
    if file:
        test = pd.read_csv(file, index_col=None)
        test.to_csv("testdata.csv", index=None)
        st.dataframe(test)
        preds = aml.predict(test)
        test.reset_index(drop=True, inplace=True)
        preds.reset_index(drop=True, inplace=True)
        findf = pd.concat([test,preds], axis=1)
        st.info(findf)




