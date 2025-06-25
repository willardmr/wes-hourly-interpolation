import datetime
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt


def read_year_data(filename):
    data = pd.read_excel(filename, sheet_name=0)
    other_data = pd.read_csv("scalars.csv")
    for i in range(1, 12):
        data = pd.concat([data, pd.read_excel(filename, sheet_name=i)])
    data = data[
        (data["Settlement Point Type"] == "HU")
        & (data["Settlement Point Name"] == "HB_HOUSTON")
        & (data["Settlement Point Price"] < 200)
    ]
    data["Delivery Date"] = pd.to_datetime(data["Delivery Date"])
    data["Month"] = data["Delivery Date"].dt.month
    data["Day"] = data["Delivery Date"].dt.day
    data = data[(data["Month"] == 8)]

    data["Year"] = data["Delivery Date"].dt.year
    data["Is Weekend"] = data["Delivery Date"].dt.dayofweek > 4
    data["Settlement Point Name"] = data["Settlement Point Name"].apply(
        lambda x: x.split("_")[1]
    )
    print(data.head())
    data["Hour"] = data["Delivery Hour"]
    other_data["Month"] = other_data["month"]
    other_data["Hour"] = other_data["delivery_hour"]
    other_data["Settlement Point Name"] = other_data["settlement_point_name"]
    other_data["Is Weekend"] = other_data["is_weekend"]

    data = data.merge(
        other_data,
        how="inner",
        on=["Month", "Hour", "Settlement Point Name", "Is Weekend"],
    )
    data = data[
        [
            "Year",
            "Month",
            "Day",
            "Hour",
            "Delivery Interval",
            "Is Weekend",
            "Settlement Point Price",
            "predicted_price",
        ]
    ]
    print(data.head())

    monthly_mean = data.groupby(["Month"]).mean().reset_index()
    monthly_mean["Average Monthly Price"] = monthly_mean["Settlement Point Price"]

    data = data.merge(
        monthly_mean[["Month", "Average Monthly Price"]],
        how="inner",
        on=["Month"],
    )
    data["predicted_price"] = data["predicted_price"] * data["Average Monthly Price"]
    data["datetime"] = pd.to_datetime(
        dict(year=data.Year, month=data.Month, day=data.Day, hour=data.Hour)
    )

    print(data.head())
    final = data.groupby(["datetime"]).mean().reset_index()

    print("HERE")
    print(final.head())
    final.plot(y=["predicted_price", "Settlement Point Price"], x="datetime")
    plt.show()


read_year_data("data/rpt.00013061.0000000000000000.RTMLZHBSPP_2024.xlsx")
