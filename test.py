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

    missing_cols,missing_series = search_missing_value(df)
    if len(missing_cols) != 0 and len(missing_series) != 0:
        df = interpolation(df,missing_cols,
                                    ['linear','linear'])  
        missing_check = 1
    else:
        missing_check = 0

    st.subheader('Missing val')
    if missing_check == 1:
        st.write(missing_cols)
        st.write(missing_series)
    else:
        st.write("not found missing val")

    st.subheader('Data header')
    st.write(df.head())

    # 시각화를 할지 말지...
    st.subheader('Data visiaulize')
    st.write("...")

    st.subheader('Setting model')
    target_ = st.selectbox('select target', df.columns, len(df.columns)-1)
    b = pre(df, target_, True, 'outliar')
    metrics_arr = b.get_metrics()
    metrics_arr = metrics_arr['Name']
    error_ = st.selectbox('select error', metrics_arr)


if start_flag:
    st.subheader('Data setup')
    with st.spinner("setting..."):
        st.write(b)
        save_df_result = save_df()

    st.subheader('Model list')
    with st.spinner("model list..."):
        st.write(b.models())
        model_list = b.models()

    st.subheader('Compare model')
    with st.spinner("compare model.."):
        all_result = compare(error_)
        st.write('Best Model (', error_, ')')
        st.write(all_result)
        save_df_result = save_df()
        st.write('Compare Model')
        st.write(save_df_result)
    
    st.subheader('Model Tuning')
    with st.spinner('tuning...'):
        tuned_model = tune(all_result, 'R2')
        tune_df = save_df()
        st.write(tune_df)
        st.write(tuned_model)

    st.subheader("Model evaluate")
    with st.spinner("model evaluate..."):
        eval_model = evaluate(tune_df)
        st.write(eval_model)

elif not data and start_flag:
    st.error('데이터를 넣어주세요')


    
# compare(error_) 보내주신 코드처럼 저렇게 나오게 하고싶은데
# 먼저 compare를 하고 사용자가 모델 선택을 할 수 있게?
# 모델을 선택하면 모델 생성(single()) / 평가지표 시각화 (singgle_visual)
# save 버튼을 통해 model 저장
# evlaute 진행 

# 버튼이 섹션 별로 나오면 좋은데..
