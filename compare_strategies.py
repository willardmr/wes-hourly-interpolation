import pandas as pd
import numpy as np
import math
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt


def read_year_data(filename):
    data = pd.read_excel(filename, sheet_name=0)
    for i in range(1, 12):
        data = pd.concat([data, pd.read_excel(filename, sheet_name=i)])
    data = data[(data["Settlement Point Name"] == "LZ_HOUSTON") & (data["Settlement Point Type"] == "LZ")]
    data['Delivery Date'] = pd.to_datetime(data['Delivery Date'])
    data["Month"] = data["Delivery Date"].dt.month
    data["Year"] = data["Delivery Date"].dt.year
    data["Is Weekend"] = data['Delivery Date'].dt.dayofweek > 4
    data = data[["Year", "Month", "Delivery Hour", "Delivery Interval", "Is Weekend", "Settlement Point Price"]]

    monthly_mean = data.groupby(['Month']).mean().reset_index()
    monthly_mean['Average Monthly Price'] = monthly_mean['Settlement Point Price']

    data = data.merge(monthly_mean[['Month', 'Average Monthly Price']], how='inner', on=['Month'])
    data['Settlement Point Price'] = data['Settlement Point Price'] - data['Average Monthly Price']

    return data.groupby(['Month', 'Delivery Hour', "Is Weekend"]).mean().reset_index()

def get_error_of_dfs(df1, df2):
    merged = df1.merge(df2, how='inner', on=['Month', "Delivery Hour", "Is Weekend"])
    merged["difference"] = merged["Settlement Point Price_x"] - merged["Settlement Point Price_y"]
    merged["difference"] = merged["difference"].abs()
    return merged[["difference"]].quantile([0.5, 0.9,0.99])

def train_model_and_evaluate(train, test):
    X_train = train[["Month", "Delivery Hour", "Is Weekend"]]
    y_train = train[["Settlement Point Price"]]

    X_test = test[["Month", "Delivery Hour", "Is Weekend"]]

    hyperparameter_grid = {
        'n_estimators': [100, 350, 400, 450],
        'max_depth': [4,56,7],
        'learning_rate': [0.001, 0.01, 0.05],
        'min_child_weight': [0.001, 0.1, 0.5, 1]
        }
    
    model = xgb.XGBRegressor()
    random_cv = RandomizedSearchCV(estimator=model,
                param_distributions=hyperparameter_grid,
                cv=5, n_iter=400,
                scoring = 'neg_mean_absolute_error',n_jobs = 4,
                return_train_score = True,
                random_state=48)

    random_cv.fit(X_train,y_train)

    reg = random_cv.best_estimator_
    y_pred = reg.predict(X_test)
    actual_values = test[["Settlement Point Price"]]
    y_pred = pd.DataFrame(y_pred, columns=['Predicted Price'])
    merged = y_pred.join(actual_values, how='inner')
    merged["difference"] = merged["Settlement Point Price"] - merged["Predicted Price"]
    merged["difference"] = merged["difference"].abs()
    return merged[["difference"]].quantile([0.5,0.9,0.99])
   

files = [
#"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2012.xlsx" ,
"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2013.xlsx" ,
"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2014.xlsx" ,
"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2015.xlsx" ,
"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2016.xlsx" ,
"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2017.xlsx" ,
"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2018.xlsx" ,
"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2019.xlsx" ,
"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2020.xlsx" ,
"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2021.xlsx" ,
"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2022.xlsx" ,
"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2023.xlsx" ,
#"data/rpt.00013061.0000000000000000.RTMLZHBSPP_2024.xlsx"
]
datas = []
for name in files:
    print(name)
    datas.append(read_year_data(name))
mlpercentile50 = []
mlpercentile90 = []
mlpercentile99 = []
brutepercentile50 = []
brutepercentile90 = []
brutepercentile99 = []
megabrutepercentile50 = []
megabrutepercentile90 = []
megabrutepercentile99 = []

for index in range(3, len(datas)):
    train = pd.concat([datas[index - 3], datas[index-2]])

    test = datas[index].copy(deep=True)
    ml_error = train_model_and_evaluate(train, test)
    brute_error = get_error_of_dfs(datas[index-2], test)
    test_copy = datas[index].copy(deep=True)
    test_copy["Settlement Point Price"] = test_copy["Average Monthly Price"]
    mega_brute_error = get_error_of_dfs(test_copy, test)

    mlpercentile50.append(ml_error["difference"].iloc[0])
    mlpercentile90.append(ml_error["difference"].iloc[1])
    mlpercentile99.append(ml_error["difference"].iloc[2])
    brutepercentile50.append(brute_error["difference"].iloc[0])
    brutepercentile90.append(brute_error["difference"].iloc[1])
    brutepercentile99.append(brute_error["difference"].iloc[2])
    megabrutepercentile50.append(mega_brute_error["difference"].iloc[0])
    megabrutepercentile90.append(mega_brute_error["difference"].iloc[1])
    megabrutepercentile99.append(mega_brute_error["difference"].iloc[2])

print("Brute")
print("50th percentile error: " + str(sum(brutepercentile50) / len(brutepercentile50)))
print("90th percentile error: " + str(sum(brutepercentile90) / len(brutepercentile90)))
print("99th percentile error: " + str(sum(brutepercentile99) / len(brutepercentile99)))
print("Mega Brute")
print("50th percentile error: " + str(sum(megabrutepercentile50) / len(megabrutepercentile50)))
print("90th percentile error: " + str(sum(megabrutepercentile90) / len(megabrutepercentile90)))
print("99th percentile error: " + str(sum(megabrutepercentile99) / len(megabrutepercentile99)))
print("ML")
print("50th percentile error: " + str(sum(mlpercentile50) / len(mlpercentile50)))
print("90th percentile error: " + str(sum(mlpercentile90) / len(mlpercentile90)))
print("99th percentile error: " + str(sum(mlpercentile99) / len(mlpercentile99)))
