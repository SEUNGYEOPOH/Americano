def pre(data,target,use_gpu,outliar):
    import pandas as pd
    import numpy as np
    import pycaret.regression 
    return pycaret.regression.setup(data, target = target, session_id = 123, train_size=0.7, use_gpu=use_gpu, remove_outliers = outliar)
def save_df():
    import pycaret.regression 
    results = pycaret.regression.pull()
    return results
def compare(standard):
    import pycaret.regression 
    return pycaret.regression.compare_models(sort = standard,)
def tune(model,opt):
    import pycaret.regression
    return pycaret.regression.tune_model(model,optimize = opt, choose_better = True)
def single(name):
    import pycaret.regression
    return pycaret.regression.create_model(name)
def single_visual(df):
    import pandas as pd
    visual = df.iloc[0:9]
    return visual.plot()
def evaluate(model):
    import pycaret.regression
    #return pycaret.regression.plot_model(model, plot = shape)
    return pycaret.regression.evaluate_model(model)
def shap(model):
    import pycaret.regression
    return pycaret.regression.interpret_model(model)
def prediction(model):
    import pycaret.regression
    return pycaret.regression.predict_model(model)
def save(model,name):
    import pycaret.regression
    return pycaret.regression.save_model(model,name)
def load(name):
    import pycaret.regression
    return pycaret.regression.load_model(name)
def search_missing_value(data):
    import pandas as pd
    col = list(data.columns)
    missing_series = data.isnull().sum()
    missing_cols = []
    for i in col:
        if missing_series[i]!=0:
            missing_cols.append(i)
        else:
            missing_series = missing_series.drop(i)
    return missing_cols,missing_series
def interpolation(data,target,method): #target = missing_cols
    import pandas as pd
    for i in range(len(target)):
        data[target[i]] = data[target[i]].interpolate(method = method[i])
    return data
    
import pandas as pd
import numpy as np
#data = pd.read_csv("./include_miss.csv")
#
#print(data)
#
#data.interpolate
#
#missing_cols,missing_series = search_missing_value(data)
#
#print(missing_cols)
#
#print(missing_series)
#
#inter_data = interpolation(data,missing_cols,['linear','linear'])
#
#print(inter_data)
#
#missing_cols,missing_series = search_missing_value(inter_data)
#
#print(missing_cols)
#
#print(missing_series)
#
#import pandas as pd
#import numpy as np
#data = pd.read_csv("./Tem_pH_Data.csv")
#
#print(data)
#
#b=pre(data,"PR",True) 
#
#b.data['pH']
#
#print(b.X) 
#print(b.y) 
#print(b.test) 
#print(b.test_transformed)
#print(b.train)
#print(b.train_transformed)
#print(b.X_test)
#print(b.X_test_transformed)
#print(b.X_train)
#print(b.X_train_transformed)
#print(b.X_test)
#print(b.X_test_transformed)
#
#save_df_result=save_df()
#
#print(save_df_result)
#
#metrics_arr = b.get_metrics()
#metrics_arr = metrics_arr['Name']
#print(metrics_arr)
#
#b.models()
#
#single_model = single('xgboost')
#
#print(single_model)
#
#a=save_df()
#b=single_visual(a)
#print(b)
#
#all_result = compare('mse')
#
#a = save_df()
#print(a)
#
#hyper = tune(all_result,'R2') #insert metrics, RandomGridSearch algorithm
#
#a = save_df()
#print(a)
#
#print(all_result)
#
#print(hyper)
#
#c = evaluate(hyper,'residuals')
#c.save
#
#print(hyper)
#
#shap(hyper)
#
#pred = prediction(hyper)
#
#sa = save(hyper, 'hyper_pipeline')
#
#load = load('hyper_pipeline')
#
#prediction(load)

data = pd.read_csv()