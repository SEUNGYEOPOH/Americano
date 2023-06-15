import streamlit as st
import AutoRegression as AR
import pandas as pd
import shap as sp
from streamlit_shap import st_shap

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("# Regression")
st.sidebar.markdown("# Regression")

r_data_ = st.file_uploader('upload regression data')

regression_flag = st.button("start")

if r_data_:
    r_data = pd.read_csv(r_data_)

    missing_cols,missing_series = AR.search_missing_value(r_data)
    if len(missing_cols) != 0 and len(missing_series) != 0:
        r_data = AR.interpolation(r_data, missing_cols,
                                ['linear','linear'])  
        missing_check = 1
    else:
        missing_check = 0

    st.subheader('Missing Value')
    if missing_check == 1:
        st.write(missing_cols)
        st.write(missing_series)
    else:
        st.write("Not Found Missing Value")

    st.subheader('Data header')
    st.write(r_data.head())

    # 시각화를 할지 말지...
    st.subheader('Data Visiaulize')
    st.write("...")

    st.subheader('Setting Model')
    target_ = st.selectbox('Select Target', r_data.columns, len(r_data.columns)-1)
    setting = AR.setup(r_data, target_, True, True)
    setting_result = AR.save_df()

    metrics_arr = setting.get_metrics()
    metrics_arr = metrics_arr['Name']
    error_ = st.selectbox('Select Error', metrics_arr)


if regression_flag:
    st.subheader('Data Setup')
    with st.spinner("setting..."):
        st.write(setting_result)

    st.subheader('Model List')
    with st.spinner("model list..."):
        model_ = setting.models()
        st.write(model_)
        best_model = model_.index[0]
        st.write('Best Model :', best_model)
        
    st.subheader('Single Model')
    with st.spinner("single model.."):
        single_model = AR.single(best_model)
        single_result = AR.save_df()
        st.write(single_result)

    with st.spinner("single visual graph data"):
        single_visual_result = AR.single_visual(single_result)
        single_visual_graph = AR.save_df()
        st.write('Single Visual Graph Data')
        st.line_chart(single_visual_graph.drop(['Mean', 'Std'], axis=0))
    

    st.subheader('Compare Model')
    with st.spinner("compare model.."):
        best_model = AR.compare(error_)
        compare_matrix = AR.save_df()
        st.write(compare_matrix)
    
    st.subheader('Model Tuning')
    with st.spinner('tuning...'):  
        optimize_best_model = AR.tune(best_model, error_)
        optimize_model_matrix = AR.save_df()
        st.write(optimize_model_matrix)

    #st.subheader("Ensemble based Soft Voting")
    #model_list = []
    #for i in range(len(model_.index)):
    #    model_list.append(model_.index[i])
    #blend_list = st.multiselect("Choose 3-Model", model_list)
    #if len(blend_list) == 3:
    #    blender = AR.Blend(blend_list)
    #    st.write(blender)

    st.subheader("Visual & Evaluate")
    with st.spinner("evaluate..."):
        AR.evaluate(optimize_best_model)

    st.write("SHAP")
    with st.spinner("shap"):
        shap = AR.shap(optimize_best_model)
        st.write(shap)
        
    st.write("Predict")
    with st.spinner("predict..."):
        pred = AR.prediction(optimize_best_model)
        st.write(pred)

    st.write("Save")
    with st.spinner("save model"):
        save_model = AR.save_model(optimize_best_model, 'pipeline')
        st.write(save_model)

    st.write("Model Load")
    with st.spinner("load model"):
        load = AR.load('pipeline')
        st.write(load)
        model_pred = AR.prediction(load)
        st.write(model_pred)
    

elif not r_data_ and regression_flag:
    st.error('데이터를 넣어주세요')