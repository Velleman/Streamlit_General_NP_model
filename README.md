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

Upload the Excel or CSV file containing your historical data. The CSV file should be comma-separated and contain two columns: ``` ds ``` with dates and ``` y ``` with numerical data. Please select the required parameters, including frequency (daily or weekly) and horizon (number of periods for prediction). It is recommended to only use advanced parameters that you understand. Click the 'Make NeuralProphet prediction' button, that appears after specifying the horizon, to generate a plot with the provided data and NeuralProphet forecast. The predicted data will also be displayed in a dataframe in the expander.
All used parameters can be found at the bottom of the page.

### Parameters

Explanation of parameters [can be found here](https://neuralprophet.com/how-to-guides/feature-guides/hyperparameter-selection.html) and [also here](https://neuralprophet.com/code/forecaster.html)

#### Required parameters

- frequency. This parameter defines how you data would be formated before sending to the NeuralProphet Machine Learning model: 
        
    ``` daily ```. The app will automatically fill in all missing dates with a one-day interval. Any missing values in the ``` y ``` column will be replaced with a value of 0;
    
    ``` weekly ```. The app will automatically fill in all missing dates with a one-day interval. Any missing values in the ``` y ``` column will be replaced with a value of 0. The data will be grouped on a weekly basis, from Monday to Sunday. The week will be marked by the date of Friday;

- horizon. The 'horizon' parameter determines the number of periods for which the user wants to make a prediction. It should be an integer greater than 0. If it is set to 0, no predictions will be made for the future.

#### Advanced parameters

- ``` normalize ```. This parameter is about scaling the time series before modelling. By default, NeuralProphet performs a (soft) min-max normalization of the time series. Normalization can help the model training process if the series values fluctuate heavily. However, if the series does not such scaling, users can turn this off or select another normalization. Possible options:

    ``` minmax ``` (Default);

    ``` off ```;

    ``` standardize ```;

    ``` soft ```;

    ``` soft1 ```.

- ``` learning_rate ```. Maximum learning rate setting for 1cycle policy scheduler (in range from 0.01 to 10). If the parameter ``` learning_rate ``` is not specified, a learning rate range test is conducted to determine the optimal learning rate. If it looks like the model is overfitting to the training data (the live loss plot can be useful hereby), reduce ``` epochs ``` and ``` learning_rate ```, and potentially increase the ``` batch_size ```. If it is underfitting, the number of ``` epochs ``` and ``` learning_rate ``` can be increased and the batch_size potentially decreased.

- ``` epochs ```. An integer between 5 and 500. Number of epochs (complete iterations over dataset) to train model. The ``` epochs ```, ``` loss_func ``` and ``` optimizer ``` are other parameters that directly affect the model training process. If not defined, ``` epochs ``` and ``` loss_func ``` are automatically set based on the dataset size.

- ```  batch_size ```. An integer between 8 and 1024.

- ``` loss_func ```. Type of loss to use. Possible options:

    ``` SmoothL1Loss ```;

    ``` MSE ```;

    ``` MAE ```.

    The default loss function is the ``` SmoothL1Loss ``` loss, which is considered to be robust to outliers.

- ``` seasonality ```. Possible options:
    
    ``` yearly_seasonality```; 
    
    ``` weekly_seasonality ```;
    
    ``` daily_seasonality ```.
    
    ``` seasonality ``` is about which seasonal components to be modelled. For example, if you use temperature data, you can probably select daily and yearly. Using number of passengers using the subway would more likely have a weekly seasonality for example. If the ``` seasonality ``` is not checked, it is at the default auto mode This lets NeuralProphet decide which of them to include depending on how much data available. For example, the yearly seasonality will not be considered if less than two years data available. In the same manner, the weekly seasonality will not be considered if less than two weeks available etc… When options are chosen the parameters are set to **True**.

- ``` seasonality_mode ```. Possible options:
        
    ``` additive ``` (Defaut);

    ``` multiplicative ```. 
    
    The default ``` seasonality_mode ``` is ``` additive ```. This means that no heteroscedasticity is expected in the series in terms of the seasonality. However, if the series contains clear variance, where the seasonal fluctuations become larger proportional to the trend, the ``` seasonality_mode ``` can be set to ``` multiplicative ```.

### Features:

- Prediction using the NeuralProphet model.

- Backtesting of predictions.

- Display of predictions in interactive plots.

- Table view of predictions including historical data.

## File Structure

- streamlit.py: The main Python script containing the Streamlit application code.

- script/script.py: Contains helper functions for data processing, modeling, and visualization.

- requirements.txt: List of Python dependencies required to run the application.

## Dependencies

- streamlit: For creating the web application interface.

- pandas: For data manipulation and analysis.

- plotly: For interactive data visualization.

- neuralprophet: For implementing the NeuralProphet model.

- scikit-learn: For backtesting and evaluation metrics.

## Notes

The code utilizes environment variables stored in a .env file for sensitive information like API tokens.

## Contributors

The app was created and maintained by **Mykola Senko** [LinkedIn](https://www.linkedin.com/in/mykola-senko-683510a4/) [GitHub](https://github.com/MykolaSenko) during the internship at [Velleman Group](https://www.velleman.eu/) (January, 3 - March 29, 2024) under the supervision of  **Bart van Audenhove** [LinkedIn](https://www.linkedin.com/in/bart-van-audenhove-2b63761b8/) with the support of the CIO of [Velleman Group](https://www.velleman.eu/) **Bennie Hebbelynck** [LinkedIn](https://www.linkedin.com/in/bennie-hebbelynck-1936a63/) and the assistance of the IT Department of [Velleman Group](https://www.velleman.eu/).

## License

This project is the property of [Velleman Group](https://www.velleman.eu/) and is intended for internal use. External use of the code is prohibited.

March 29, 2024
Gavere, Belgium