import streamlit as st
import pandas as pd
from datetime import datetime
import os

# from script.script import


st.set_page_config(layout="wide")

st.title("NeuralProphet prediction")

st.markdown(
    f"""
    This application is designed to generate time-series predictions using NeuralProphet. To make a prediction, you must upload an Excel or CSV file containing two columns: 'ds' for the time-series and 'y' for the target data you wish to predict. At the top of the page, select the prediction horizon and frequency (daily or weekly) you desire. Once you have uploaded your data and made your selections, click the 'Make Prediction' button.
    """
)

df = st.file_uploader('File uploader')

