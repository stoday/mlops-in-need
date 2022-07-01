
import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tpot import TPOTRegressor
from algorithms import automl_tpot
from sklearn.model_selection import train_test_split
import threading
import ctypes
import time
import re
import os
import logging

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
  
# 用物件執行 TPOT
class tpot_worker(threading.Thread):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0
        self.should_stop = threading.Event()
        self.session = None
        
    def run(self):
        X_dataset = dataset.iloc[:, 0:-1]
        y_dataset = dataset.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset)
        
        tpot = TPOTRegressor(generations=1, 
                             population_size=10, 
                             verbosity=2, 
                             random_state=42,
                             n_jobs=-1, 
                             log_file='./Models/log.txt')
        
        tpot.fit(X_train, y_train)
        model_pipline = tpot.export()
        self.session = model_pipline
    
    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id

        for id, thread in threading._active.items():
            if thread is self:
                return id
        
    def raise_exception(self):
            thread_id = self.get_id()
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                thread_id,
                ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
                print('Exception raise failure')     
    
  
# 讀取 TPOT 進度
def read_tpot_progress(target_log_file):
    progress_status = 0
    if os.path.isfile(target_log_file):
        try:
            log_ctx = open(target_log_file).read()
            log_ctx = log_ctx.replace('\n', '')
            all_records = re.findall('(\d+)%', log_ctx)
            
            if len(all_records) != 0:
                progress_status = np.max([int(x.replace('%', '')) for x in all_records if len(x) > 0]).tolist()
            else:
                progress_status = 0
            
        except Exception as err:
            progress = 0
        
    return progress_status


# 處理 TPOT 的輸出
def process_model_code(model_pipline):
    print(model_pipline)    # check
    flag = 0
    model_code = []
    for x in model_pipline.split('\n'):
        if 'import' in x:
            model_code.append(x) 
            
        if 'exported_pipeline.fit' in x:
            flag = 0
        
        if 'exported_pipeline = ' in x:
            model_code.append(x)
            flag = 1
        
        elif flag == 1:
            model_code.append(x)

    # print(model_code)
    model_code = '\n'.join(model_code)
    print(model_code)
    # exec(model_code)
    
    return model_code


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
            
            cols = st.columns((1, 2, 10))
            go_btn_place = cols[0].empty()
            stop_btn_place = cols[1].empty()
            
            # 按下按鈕開始執行
            if go_btn_place.button('Start', disabled=worker is not None):     
                # worker = st.session_state.worker = tpot_worker(daemon=True)
                worker = st.session_state.worker = automl_tpot.tpot_worker(dataset=dataset, daemon=True)
                worker.start()
                st.experimental_rerun()
            
            # 按下停止結束
            if stop_btn_place.button('Stop worker', disabled=worker is None):
                worker.should_stop.set()
                worker = st.session_state.worker = None
                st.experimental_rerun()
                
            if worker is None:
                st.markdown('No worker running.')
                
            else:
                placeholder = st.empty()
                prog_bar = placeholder.progress(0)
                
                # 讀取 TPOT 進度
                while worker.is_alive():                    
                    time.sleep(1)
                    progress_status = read_tpot_progress('./Models/log.txt')
                    prog_bar.progress(progress_status)
                
                model_code = worker.session
                print(process_model_code(model_code))
                worker.should_stop.set()
                worker.join()
                worker = st.session_state.worker = None
                
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
