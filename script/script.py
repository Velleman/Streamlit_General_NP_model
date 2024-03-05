import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, GridOptionsBuilder
from sklearn.metrics import mean_absolute_error
import numpy as np
from neuralprophet import NeuralProphet, set_log_level


def values_to_list(df, column, start_date, latest_date):
    df_copy = df.copy()
    df_copy["ds"] = pd.to_datetime(df_copy["ds"])
    mask = (df_copy["ds"] >= start_date) & (df_copy["ds"] <= latest_date)
    df_actual = df_copy.loc[mask, column]
    values_list = df_actual.tolist()
    return values_list


def plot_data(df):
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

def np_model(df, horizon):
    set_log_level("ERROR")
    m = NeuralProphet()  # Creating NeuralProphet model
    m.fit(df, minimal=True)  # Fitting the model to the data
    future = m.make_future_dataframe(
        df, n_historic_predictions=True, periods=horizon
    )  # Making predictions
    df_forecast = m.predict(future)
    return df_forecast


def normal_df(df, frequency):
    df["ds"] = pd.to_datetime(df["ds"])
    all_dates = pd.date_range(
        start=df["ds"].min(), end=df["ds"].max(), freq="D"
    )
    df_date = pd.DataFrame({"ds": all_dates})
    df_date["ds"] = pd.to_datetime(df_date["ds"])
    df_model = pd.merge(df_date, df, on="ds", how="left")
    df_model["y"] = df_model["y"].fillna(0)
    #df_model.loc[df_model["y"] < 0, "y"] = 0
    if frequency == "weekly":
        df_model["ds"] = df_model["ds"] + pd.to_timedelta(
            (4 - df_model["ds"].dt.dayofweek) % 7, unit="D"
        )
        df_model = df_model.groupby("ds").agg({"y": "sum"}).reset_index()
    return df_model