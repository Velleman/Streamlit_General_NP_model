import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, GridOptionsBuilder
from sklearn.metrics import mean_absolute_error
import numpy as np
from neuralprophet import NeuralProphet, set_log_level


def values_to_list(df, column, start_date, latest_date):
    """
    The function `values_to_list` extracts values from a specified column in a DataFrame within a given
    date range and returns them as a list.
    
    :param df: DataFrame containing the data
    :param column: The `column` parameter in the `values_to_list` function refers to the column in the
    DataFrame `df` from which you want to extract values. When calling the function, you would provide
    the name of the column as a string. For example, if you have a DataFrame `df` with
    :param start_date: Start date is the beginning date from which you want to filter the data
    :param latest_date: The `latest_date` parameter is used to specify the end date for filtering the
    data. Only the data within the range from `start_date` to `latest_date` will be included in the
    final list of values extracted from the specified column of the DataFrame
    :return: The function `values_to_list` returns a list of values from the specified column in the
    DataFrame `df` that fall within the date range from `start_date` to `latest_date`.
    """
    df_copy = df.copy()
    df_copy["ds"] = pd.to_datetime(df_copy["ds"])
    mask = (df_copy["ds"] >= start_date) & (df_copy["ds"] <= latest_date)
    df_actual = df_copy.loc[mask, column]
    values_list = df_actual.tolist()
    return values_list


def plot_data(df):
    """
    The function `plot_data` generates a plot comparing real data with predicted data using markers and
    lines, respectively.
    
    :param df: The `df` parameter in the `plot_data` function is expected to be a DataFrame containing
    the following columns:
    :return: The function `plot_data(df)` is returning a Plotly Figure object that contains two traces:
    one for the real data (displayed as markers in black) and one for the predicted data (displayed as
    lines in blue). The layout of the figure includes a title "Real vs Predicted Data" and axis titles
    for both the x-axis ("Date") and y-axis ("Value").
    """
    real_data = go.Scatter(
        x=df["ds"],
        y=df["y"],
        mode="markers",
        marker=dict(color="black"),
        name="Real Data",
    )

    predicted_data = go.Scatter(
        x=df["ds"],
        y=df["yhat1"],
        mode="lines",
        marker=dict(color="blue"),
        name="NeuralProphet Predicted Data",
    )

    fig = go.Figure(data=[real_data, predicted_data])

    fig.update_layout(
        title=f"Real vs Predicted Data",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Value"),
    )
    return fig


def show_table(df_sku_show):
    """
    The function `show_table` generates an interactive grid table using the provided DataFrame
    `df_sku_show` in Python.

    :param df_sku_show: The function `show_table` takes a DataFrame `df_sku_show` as input and displays
    it using the AgGrid library. The parameters used in the function are as follows:
    """
    gb = GridOptionsBuilder.from_dataframe(df_sku_show)
    gridOptions = gb.build()
    st.markdown(
        """
        <style>
        .fullWidth {
            width: 100% !important;
            hight: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    ag = AgGrid(
        data=df_sku_show,
        gridOptions=gridOptions,
        width_mode="full",
        height_mode="full",
        update_mode=GridUpdateMode.VALUE_CHANGED,
        fit_columns_on_grid_load=True,
    )


def norm_mae(df, freq):
    """
    The function calculates the normalized mean absolute error (MAE) between actual and predicted values
    in a dataframe based on a specified frequency.
    
    :param df: It seems like you were about to provide some information about the `df` parameter in the
    `norm_mae` function, but the information is missing. Could you please provide more details or
    specify what kind of information you need regarding the `df` parameter?
    :param freq: The `freq` parameter in the `norm_mae` function is used to specify the frequency of the
    data. It can take two possible values: 'weekly' or any other value. If 'weekly' is passed, the
    function will calculate the start date based on a weekly frequency (52 weeks
    :return: The function `norm_mae` returns the normalized mean absolute error (normal_mae) calculated
    based on the input DataFrame `df` and frequency `freq`.
    """
    latest_date = df.loc[df["y"].notnull(
    ), "ds"].max()
    if freq == 'weekly':
        start_date = latest_date - pd.DateOffset(weeks=52)
    else:
        start_date = latest_date - pd.DateOffset(days=52)
    list_actual = values_to_list(df, "y", start_date, latest_date)
    list_pred = values_to_list(df, "yhat1", start_date, latest_date)
    array_actual = np.array(list_actual)
    array_pred = np.array(list_pred)
    mae = mean_absolute_error(array_actual, array_pred)
    normal_mae = mae / np.sum(array_actual)
    return normal_mae

def np_model(df, horizon, **kwargs):
    """
    The function `np_model` fits a NeuralProphet model to a given dataframe and makes predictions for a
    specified horizon.
    
    :param df: It looks like the code snippet you provided is a function that uses the NeuralProphet
    library to create a time series forecasting model. The function takes a dataframe `df` containing
    the time series data, a `horizon` parameter specifying the number of periods into the future to
    forecast, and additional keyword
    :param horizon: The `horizon` parameter in the `np_model` function represents the number of future
    time steps you want to forecast. It determines how far into the future the model will make
    predictions
    :return: The function `np_model` returns the forecasted values generated by the NeuralProphet model
    for the specified horizon.
    """
    set_log_level("ERROR")
    if kwargs is not None:
        m = NeuralProphet(**kwargs)  # Creating NeuralProphet model
    else:
        m = NeuralProphet()
    m.fit(df, minimal=True)  # Fitting the model to the data
    future = m.make_future_dataframe(
        df, n_historic_predictions=True, periods=horizon
    )  # Making predictions
    df_forecast = m.predict(future)
    return df_forecast


def normal_df(df, frequency):
    """
    The function `normal_df` takes a DataFrame with a date column and a frequency parameter, fills
    missing dates with zeros, and aggregates data based on the specified frequency (daily or weekly).
    
    :param df: It seems like you were about to provide some information about the `df` parameter, but
    the information is missing. Could you please provide more details or specify what kind of
    information you need regarding the `df` parameter?
    :param frequency: The `frequency` parameter in the `normal_df` function is used to specify how the
    data should be aggregated. In this case, if the frequency is set to "weekly", the function will
    aggregate the data on a weekly basis by summing up the values for each week
    :return: The function `normal_df` returns a DataFrame `df_model` that has been processed based on
    the input parameters `df` and `frequency`. The DataFrame `df_model` is created by merging the input
    DataFrame `df` with a new DataFrame `df_date` that contains all dates within the range of dates
    present in the input DataFrame. The "y" column in the merged DataFrame is filled with
    """
    df["ds"] = pd.to_datetime(df["ds"])
    all_dates = pd.date_range(
        start=df["ds"].min(), end=df["ds"].max(), freq="D"
    )
    df_date = pd.DataFrame({"ds": all_dates})
    df_date["ds"] = pd.to_datetime(df_date["ds"])
    df_model = pd.merge(df_date, df, on="ds", how="left")
    df_model["y"] = df_model["y"].fillna(0)
    if frequency == "weekly":
        df_model["ds"] = df_model["ds"] + pd.to_timedelta(
            (4 - df_model["ds"].dt.dayofweek) % 7, unit="D"
        )
        df_model = df_model.groupby("ds").agg({"y": "sum"}).reset_index()
    return df_model