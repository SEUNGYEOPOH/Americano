import pandas as pd

def search_missing_value(data):
    col = list(data.columns)
    missing_series = data.isnull().sum()
    missing_cols = []
    for i in col:
        if missing_series[i] != 0:
            missing_cols.append(i)
        else:
            missing_series = missing_series.drop(i)
    return missing_cols, missing_series

def interpolation(data, target, method):
    for i in range(len(target)):
        data[target[i]] = data[target[i]].interpolate(method=method[i])
    return data
