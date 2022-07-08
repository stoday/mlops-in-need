
from cmath import exp
import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tpot import TPOTRegressor
from algorithms import automl_tpot
from algorithms import tpot_progress
from sklearn.model_selection import train_test_split
import threading
import ctypes
import time
import re
import os
import logging
import mlflow
from mlflow_utils import experiments

# 紀錄
logging.basicConfig(filename='app.log', 
                    filemode='w', 
                    format='%(asctime)s - %(message)s', 
                    level=logging.INFO)

# 網頁操作流程控制（實驗性質）
class controller():
    def __init__(self):
        self.example = 0

# 網頁以寬畫面呈現
st.set_page_config(layout="wide")

# 網頁標題
st.title('MLOPSIN')
# st.subheader('Running MLOps in an easy way!')

# st.session_state 初始化
if 'worker' not in st.session_state:
    st.session_state.worker = None
worker = st.session_state.worker
    
if 'model_pipline' not in st.session_state:
    st.session_state.model_pipline = None

# 側欄
with st.sidebar:
    selected = option_menu("Menu", ["Home", 'Settings'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=0)

# 橫欄
if selected == 'Home':
    process = option_menu(None, ['Import Dataset', 'Training', 'Deployment', 'Monitoring'], 
        icons=['cloud-upload', 'robot', 'brightness-alt-high', 'graph-up-arrow'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    # 上傳資料集
    if process == 'Import Dataset':
        # 如果之前沒有上傳過任何資料
        import_file = st.file_uploader("Upload a CSV File", type=['csv'])
        
        # 偵測到上傳了新資料，則新資料會存在 session_state 裡面
        if import_file is not None:
            dataset = pd.read_csv(import_file, encoding='utf-8')
            st.session_state['dataset'] = dataset
        
        if 'dataset' in st.session_state.keys():
            AgGrid(st.session_state['dataset'].head(100),
                   editable=True, 
                   height=300)
            
            st.button('Data Profile')
    
    # 模型訓練
    elif process == 'Training':
        # 如果已經有資料上傳
        if 'dataset' in st.session_state.keys():
            dataset = st.session_state['dataset']
            
            experiment_name = st.text_input('Name of the Experiment')
            if experiment_name != '':
                try:
                    experiment_id = mlflow.create_experiment(experiment_name)
                except Exception as err:
                    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            else:
                experiment_id = '-1'
            
            cols = st.columns((1, 2, 10))
            train_btn_place = cols[0].empty()
            stop_btn_place = cols[1].empty()
            
            # 按下按鈕開始執行
            if train_btn_place.button('Start', disabled=worker is not None):     
                # worker = st.session_state.worker = tpot_worker(daemon=True)
                worker = st.session_state.worker = automl_tpot.tpot_worker(dataset=dataset, daemon=True)
                worker.start()
                st.write(worker.is_alive())
                st.experimental_rerun()
            
            # 按下停止結束
            if stop_btn_place.button('Stop worker', disabled=worker is None):
                worker.should_stop.set()
                worker = st.session_state.worker = None
                st.experimental_rerun()
                
            if worker is None:
                st.markdown('No worker running.')                
                exp_results = experiments.list_all(str(experiment_id))
                # AgGrid(exp_results)
                
            else:
                placeholder = st.empty()
                prog_bar = placeholder.progress(0)
                
                # 讀取 TPOT 進度
                while worker.is_alive():                    
                    time.sleep(1)
                    progress_status = tpot_progress.read_tpot_progress('./Models/log.txt')
                    prog_bar.progress(progress_status)
                print('worker.is_alive: ' + str(worker.is_alive()) + '\n')
                model_code = worker.session
                print('\n*** OUTPUT MODEL CODE ***')
                print(model_code)
                print(tpot_progress.process_model_code(model_code))
                print('*** END ***\n')
                worker.should_stop.set()
                worker.join()
                worker = st.session_state.worker = None
                
                # 使用「找到較好的」參數進行最後模型訓練
                exported_pipeline = None
                exec(tpot_progress.process_model_code(model_code))

                X = dataset.iloc[:, :-1]
                y = dataset.iloc[:, -1]
                    
                with mlflow.start_run(experiment_id=experiment_id) as run:
                    mlflow.sklearn.autolog()
                    exported_pipeline.fit(X, y)
                    metrics = mlflow.sklearn.eval_and_log_metrics(exported_pipeline, 
                                                                  X, 
                                                                  y, 
                                                                  prefix="val_")
                    print(metrics)
                mlflow.end_run()
                print('*** mlflow done ***\n')
                
                st.experimental_rerun()
                
            st.markdown('---')
            
        else:
            st.write('No data.')
        
    
    elif process == 'Deployment':
        st.markdown('---')        
        cols = st.columns((1, 3, 2, 2))
        cols[0].write('**Index**')
        cols[1].write('**Model Name**')
        cols[2].write('**Score**')
        cols[3].write('**Deploy**')

        cols = st.columns((1, 3, 2, 2))
        cols[0].write('1')
        cols[1].write('1st-examples')
        cols[2].write(0.92)
        phold_1 = cols[3].empty()
        btn_1 = phold_1.button('Deploy', key='1') 
        
        cols = st.columns((1, 3, 2, 2))
        cols[0].write('2')
        cols[1].write('2nd-examples')
        cols[2].write(0.98)
        phold_2 = cols[3].empty()
        btn_2 = phold_2.button('Deploy', key='2')    
        
        cols = st.columns((1, 3, 2, 2))
        cols[0].write('3')
        cols[1].write('3rd-examples')
        cols[2].write(0.82)
        phold_3 = cols[3].empty()
        btn_3 = phold_3.button('Deploy', key='3')      
                    
        if btn_1:
            phold_1.button('DEPLOYED')

        if btn_2:
            phold_2.button('DEPLOYED')
            
        if btn_3:
            phold_3.button('DEPLOYED')
        
        st.markdown('---')
    
    elif process == 'Monitoring':
        data = np.exp(np.arange(0, 1, step=0.01))
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(data, 'r')
        ax.scatter(np.arange(0, 100, step=1), data+np.random.randn(100)/10)
        ax.set_title('Performace')
        ax.set_ylabel('NG Rate (%)')
        ax.set_xlabel('Batch Index')
        ax.grid()
        
        plot_spec = st.columns((1, 8, 1))
        plot_spec[1].pyplot(fig)
        
        
else:
    st.write('hello')
