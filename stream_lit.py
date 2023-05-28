def pre(data,target,use_gpu,outliar):
    import pandas as pd
    import numpy as np
    import pycaret.regression 
    return pycaret.regression.setup(data, target = target, session_id = 123, train_size=0.7, 
                                    use_gpu=use_gpu, remove_outliers = outliar)
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

import streamlit as st
import numpy as np
import pandas as pd

st.title("Americano")
st.header("Term-project")

data = st.file_uploader('upload data')

start_flag = st.button("start")

if data:
    df = pd.read_csv(data)

    if start_flag:
        st.subheader('data header')
        st.write(df.head())

        missing_cols,missing_series = search_missing_value(df)

        st.subheader('Missing val')
        if len(missing_cols) != 0 and len(missing_series) != 0:
            st.write(missing_cols)
            st.write(missing_series)
            inter_data = interpolation(df,missing_cols,
                                       ['linear','linear'])
                                       
        else:
            st.write("not found missing val")

        st.subheader('Data visiaulize')
        st.write("...")

        st.subheader('Setting model')
        target_ = st.selectbox('select target', df.columns, len(df.columns)-1)
        error_ = st.selectbox('select error', ['MAE','MSE','RMSE','R2','RMSLE','MAPE'])

        model_flag = st.button("model make")
        
        if model_flag:
            b = pre(df, target_, True, 'outliar')
            with st.spinner("진행중.."):
                all_result = compare(error_)
            all_result
            save_df_result = save_df()
            b.models()


        #hyper = tune(all_result,'R2')


        #st.subheader('model list')
        #st.write(b.models())

elif not data and start_flag:
    st.error('데이터를 넣어주세요')

    
#col1, col2, col3 = st.columns(3)
#with col1:
#    st.write(' ')
#with col2:
#    st.image("tuk.png")
#with col3:
#    st.write(' ')