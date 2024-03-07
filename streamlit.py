import streamlit as st
import pandas as pd

from script.script import show_table, normal_df, np_model, plot_data, norm_mae


st.set_page_config(layout="wide")

st.title("NeuralProphet prediction")

st.markdown(
    f"""
    This application is designed to generate time-series predictions using NeuralProphet. To make a prediction, you must upload an Excel or CSV file containing two columns: 'ds' for the time-series and 'y' for the target data you wish to predict. At the top of the page, select the prediction horizon and frequency (daily or weekly) you desire. Once you have uploaded your data and made your selections, click the 'Make Prediction' button.
    """
)

uploaded_file = st.file_uploader('File uploader', type=["csv", "xlsx"])

# This block of code is responsible for reading the uploaded file and displaying its contents in a
# table within a Streamlit expander.
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
# `parameters = {}` is initializing an empty dictionary named `parameters`. This dictionary is used to
# store advanced parameters that the user can select when making predictions using the NeuralProphet
# model. The parameters are set based on the user's selections in the Streamlit interface, and they
# are then passed as keyword arguments to the `np_model` function for making predictions. This allows
# the user to customize the prediction process by specifying parameters such as normalization type,
# learning rate, epochs, batch size, loss function, seasonality, and seasonality mode.
parameters = {}

col1, col2 = st.columns(2)
# The code block `with col1:` is creating a column layout in the Streamlit interface where the
# subsequent elements will be placed. Within this column layout, the code is displaying a radio button
# using `st.radio` with the prompt 'Choose frequency:' and options 'daily' and 'weekly'.
with col1:
    freq = st.radio('Choose frequency:', ['daily','weekly'])
    st.write(f"You have chosen frequency: {freq}")

# The code block `with col2:` is creating a column layout in the Streamlit interface where the
# subsequent elements will be placed. Within this column layout, the code is displaying a number input
# field using `st.number_input` with the prompt 'Please specify the number of periods for which you
# would like to make a prediction:', a minimum value of 0, and a step of 1. This input field allows
# the user to specify the number of periods for which they would like to make a prediction using the
# NeuralProphet model. The value entered by the user in this input field will be stored in the
# variable `horiz`, which represents the prediction horizon.
with col2:
    horiz = st.number_input('Please specify the number of periods for which you would like to make a prediction:', min_value=0, step=1)
    
# The code block you provided is creating an expander in the Streamlit interface titled "Choose
# Advanced Parameters" with a link to a guide on hyperparameter selection for NeuralProphet. Within
# this expander, there are several interactive elements that allow the user to customize advanced
# parameters for making predictions using the NeuralProphet model.
with st.expander("Choose [Advanced Parameters](https://neuralprophet.com/how-to-guides/feature-guides/hyperparameter-selection.html)"):
    select_normal = st.checkbox("Check to select [normalize parameter](https://neuralprophet.com/code/forecaster.html)")
    if select_normal:
        normalize = st.selectbox("Choose the type of normalization to apply to the time series:", ['minmax (Default)', 'off', 'standardize', 'soft', 'soft1'])
        if normalize == 'minmax (Default)':
            normalize = 'minmax'
        parameters['normalize'] = normalize
    select_learn = st.checkbox("Check to select [learning rate parameter](https://neuralprophet.com/code/forecaster.html). Auto by default.")
    if select_learn:
        learning_rate = st.number_input('Enter a learning rate between 0.01 and 10', min_value=0.01, max_value=10.00, step=0.01)
        parameters['learning_rate'] = learning_rate
    select_epochs = st.checkbox("Check to select [epochs parameter](https://neuralprophet.com/code/forecaster.html). Auto by default.")
    if select_epochs:
        epochs = st.number_input('Enter a number of epochs between 5 and 500', min_value=5, max_value=500, step=1)
        parameters['epochs'] = epochs
    select_batch_size = st.checkbox('Check to select [batch size parameter](https://neuralprophet.com/code/forecaster.html). Auto by default.')
    if select_batch_size:
        batch_size = st.number_input('Enter a batch size between 8 and 1024', min_value=8, max_value=1024, step=1)
        parameters['batch_size'] = batch_size
    select_loss = st.checkbox("Check to select [loss function parameter](https://neuralprophet.com/code/forecaster.html)")
    if select_loss:
        loss_func = st.selectbox("Choose the type of loss function parameter:", ['SmoothL1Loss (Default)', 'MSE', 'MAE'])
        if loss_func == 'SmoothL1Loss (Default)':
            loss_func = 'SmoothL1Loss'
        parameters['loss_func'] = loss_func
    select_seasonality = st.checkbox("Check to select [seasonality parameter](https://neuralprophet.com/code/forecaster.html). Auto be default.")
    if select_seasonality:
        seasonality_options = ['yearly_seasonality', 'weekly_seasonality', 'daily_seasonality']
        selected_seasonality = st.multiselect("Choose the type of seasonality parameter:", seasonality_options)
        if 'yearly_seasonality' in selected_seasonality:
            parameters['yearly_seasonality'] = True
        if 'weekly_seasonality' in selected_seasonality:
            parameters['weekly_seasonality'] = True
        if 'daily_seasonality' in selected_seasonality:
            parameters['daily_seasonality'] = True
    select_seasonality_mode = st.checkbox("Check to select [seasonality mode parameter](https://neuralprophet.com/code/forecaster.html)")
    if select_seasonality_mode:
        seasonality_mode = st.selectbox("Choose the type of loss function parameter:", ['additive (Default)', 'multiplicative'])
        if seasonality_mode == 'additive (Default)':
            seasonality_mode = 'additive'
        parameters['seasonality_mode'] = seasonality_mode

# This code snippet is checking if the variable `horiz` is not None and if its value is greater than
# 0. If both conditions are met, it will display a button labeled 'Make NeuralProphet prediction'
# using Streamlit's `st.button` function. This button allows the user to trigger the process of making
# predictions using the NeuralProphet model based on the uploaded data and selected parameters.
if horiz is not None and horiz > 0:
    make_prediction = st.button('Make NeuralProphet prediction')

    # This block of code is responsible for handling the process of making predictions using the
    # NeuralProphet model when the user clicks the 'Make NeuralProphet prediction' button in the
    # Streamlit interface. Here's a breakdown of what each part of the code is doing:
    if make_prediction:
        with st.spinner(text='In progress'):
            try:
                df_model = normal_df(df, freq)
                st.success('Done')
            except Exception as e:
                st.warning(" Please upload the valid file", icon="⚠️")
                pass
            df_pred = np_model(df_model, horiz, **parameters)
            normal_mae = norm_mae(df_pred, freq)
        try:
            fig = plot_data(df_pred)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(
                f"Normalized Mean Absolute Error for the prediction:"
            )
            st.info(normal_mae)
        
            with st.expander("Show the table with predicted values"):
                show_table(df_pred)
            st.write("Prediction was made with these parameters:")
            st.write(f"Freaquency: {freq}.")
            st.write(f"Horizon: {horiz} periods.")
            if select_normal:
                st.write(f"Advanced parameter: Type of Normalization: {normalize}.")
            if  select_learn:
                st.write(f"Advanced parameter: Learning Rate: {learning_rate}.")
            if select_epochs:
                st.write(f"Advanced parameter: Number of Epochs: {epochs}.")
            if select_batch_size:
                st.write(f"Advanced parameter: Batch Size: {batch_size}.")
            if select_loss:
                st.write(f"Advanced parameter: Type of Loss Function: {loss_func}.")
            if select_seasonality:
                st.write(f"Advanced parameter: Type of Seasonality: {selected_seasonality}.")
            if select_seasonality_mode:
                st.write(f"Advanced parameter: Seasonality mode: {seasonality_mode}.")
        except Exception as e:
            st.warning("You have not uploaded a file with data or uploaded file is not supported.", icon="⚠️")
            pass