import pandas as pd
import numpy as np
import pycaret.regression

def setup(data, target, use_gpu, outliar):
    return pycaret.regression.setup(data, target=target, session_id=123, train_size=0.7, use_gpu=use_gpu, remove_outliers=outliar)

def save_df():
    results = pycaret.regression.pull()
    return results

def compare(standard):
    return pycaret.regression.compare_models(sort=standard)

def tune(model, opt):
    return pycaret.regression.tune_model(model, optimize=opt, choose_better=True)

def Blend(arr):
    arr[0]=pycaret.regression.create_model(arr[0])
    arr[1]=pycaret.regression.create_model(arr[1])
    arr[2]=pycaret.regression.create_model(arr[2])
    return pycaret.regression.blend_models([arr[0],arr[1],arr[2]])
def single(name):
    return pycaret.regression.create_model(name)

def single_visual(df):
    visual = df.iloc[0:9]
    return visual.plot()

def evaluate(model):
    return pycaret.regression.evaluate_model(model)

def shap(model):
    return pycaret.regression.interpret_model(model)

def prediction(model):
    return pycaret.regression.predict_model(model)

def save_model(model, name):
    return pycaret.regression.save_model(model, name)

def load(name):
    return pycaret.regression.load_model(name)


