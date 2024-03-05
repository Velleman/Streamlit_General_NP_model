import streamlit as st
import pandas as pd
from datetime import datetime
import os

from script.script import show_table, normal_df, np_model, plot_data, norm_mae


st.set_page_config(layout="wide")

st.title("NeuralProphet prediction")

st.markdown(
    f"""
    This application is designed to generate time-series predictions using NeuralProphet. To make a prediction, you must upload an Excel or CSV file containing two columns: 'ds' for the time-series and 'y' for the target data you wish to predict. At the top of the page, select the prediction horizon and frequency (daily or weekly) you desire. Once you have uploaded your data and made your selections, click the 'Make Prediction' button.
    """
)

uploaded_file = st.file_uploader('File uploader', type=["csv", "xlsx"])

if uploaded_file is not None:
    # Determine the file type
    if uploaded_file.name.endswith('.csv'):
        # Read CSV file
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        # Read Excel file
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    with st.expander("Show the table"):
        show_table(df)

col1, col2 = st.columns(2)
with col1:
    freq = st.radio('Choose frequency:', ['daily','weekly'])
    st.write(f"You have chosen frequency: {freq}")

with col2:
    horiz = st.number_input('Please specify the number of periods for which you would like to make a prediction:', min_value=0, step=1)
    

if horiz is not None and horiz > 0:
    make_prediction = st.button('Make NeuralProphet prediction')

    if make_prediction:
        with st.spinner(text='In progress'):
            df_model = normal_df(df, freq)
            df_pred = np_model(df_model, horiz)
            normal_mae = norm_mae(df_pred, freq)
            st.success('Done')

        fig = plot_data(df_pred)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(
            f"Normalized Mean Absolute Error for the prediction:"
        )
        st.info(normal_mae)
    
        with st.expander("Show the table with predicted values"):
            show_table(df_pred)