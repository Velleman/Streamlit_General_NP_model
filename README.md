# Streamlit app for NeuralProphet predictions.

This Streamlit application uses NeuralProphet to make predictions based on your uploaded data. [NeuralProphet](https://github.com/ourownstory/neural_prophet/blob/main/README.md) is a user-friendly framework for interpretable time series forecasting that combines neural networks and traditional time-series algorithms. It is built on PyTorch and inspired by Facebook Prophet and AR-Net. Although the Streamlit app does not use neural networks for prediction (``` n_lag ``` parameter is set on ```0```), the user can adjust various parameters.

## Getting Started
To run this Streamlit app locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using the following command:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app using the following command:

   ```python
   streamlit run streamlit.py
   ```
If you want to run the app in Docker container, follow these steps:

2. Build a docker image using the following command:

   ```bash
   docker build -t streamlit .
   ```
3. Run the Docker container using the this command:

   ```bash
   docker run -p 8502:8502 streamlit
   ```

## Usage of the app and features

Upload the Excel or CSV file containing your historical data. The CSV file should be comma-separated and contain two columns: ``` ds ``` with dates and ``` y ``` with numerical data. Please select the required parameters, including frequency (daily or weekly) and horizon (number of periods for prediction). It is recommended to only use advanced parameters that you understand. Click the 'Make NeuralProphet prediction' button to generate a plot with the provided data and NeuralProphet forecast. The predicted data will also be displayed in a dataframe in the expander.
All used parameters can be found at the bottom of the page.

### Parameters

Explanation of parameters [can be found here.](https://neuralprophet.com/how-to-guides/feature-guides/hyperparameter-selection.html)

#### Required parameters

- frequency. This parameter defines how you data would be formated before sending to the NeuralProphet Machine Learning model: 
        
    ``` daily ```. The app will automatically fill in all missing dates with a one-day interval. Any missing values in the ``` y ``` column will be replaced with a value of 0;
    ``` weekly ```. The app will automatically fill in all missing dates with a one-day interval. Any missing values in the ``` y ``` column will be replaced with a value of 0. The data will be grouped on a weekly basis, from Monday to Sunday. The week will be marked by the date of Friday;

- horizon. The 'horizon' parameter determines the number of periods for which the user wants to make a prediction. It should be an integer greater than 0. If it is set to 0, no predictions will be made for the future.

#### Advanced parameters
