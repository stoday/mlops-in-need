
import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
import numpy as np
import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from streamlit.scriptrunner import add_script_run_ctx, StopException
import threading
import time
import re
import os
import logging

logging.basicConfig(filename='app.log', 
                    filemode='w', 
                    format='%(asctime)s - %(message)s', 
                    level=logging.INFO)


class controller():
    def __init__(self):
        self.example = 0

st.set_page_config(layout="wide")

st.title('MLOPSIN')
st.subheader('Running MLOps in an easy way!')
# st.session_state['dataset'] = None

# 執行 TPOT
def tpot_work(dataset):
    st.write('training...')
    X_dataset = dataset.iloc[:, 0:-1]
    y_dataset = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset)
    
    if os.path.isfile('./Models/log.txt'):
        os.remove('./Models/log.txt')
        print('remove model log')
    
    tpot = TPOTRegressor(generations=1, 
                         population_size=50, 
                         verbosity=2, 
                         random_state=42, 
                         log_file='./Models/log.txt')
    
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_boston_pipeline.py')
    
    if os.path.isfile('./Models/log.txt'):
        os.remove('./Models/log.txt')
        print('remove model log')
    
  
# 讀取 TPOT 進度
def read_tpot_progress(target_log_file):
    if os.path.isfile(target_log_file):
        log_ctx = open(target_log_file).read()
        log_ctx = log_ctx.replace('\n', '')
        all_records = re.findall('(\d+)%', log_ctx)
        if len(all_records) != 0:
            progress_status = np.max([int(x.replace('%', '')) for x in all_records if len(x) > 0]).tolist()
        else:
            progress_status = 0
        
    else:
        progress_status = 0
        
    return progress_status


# 側欄
with st.sidebar:
    selected = option_menu("Menu", ["Home", 'Settings'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=0)
    '<' + selected + '>'

# 橫欄
if selected == 'Home':
    process = option_menu(None, ['Import Dataset', 'Training', 'Deployment', 'Monitoring'], 
        icons=['cloud-upload', 'robot', 'brightness-alt-high', 'graph-up-arrow'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    '<' + process + '>'
    
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
        else:
            st.write('No data.')
    
    # 模型訓練
    elif process == 'Training':
        dataset = st.session_state['dataset']
        if (st.button('Start')) and (dataset is not None):
            # 開一個新的執行續來跑 TPOT 
            thread = threading.Thread(target=tpot_work, args=(dataset,))
            add_script_run_ctx(thread)
            thread.start()
            
            # 顯示執行 TPOT 的進度
            st.write('Processing...')
            prog_bar = st.progress(0)
            progress_status = 0
            while progress_status != 100:
                time.sleep(1)
                progress_status = read_tpot_progress('./Models/log.txt')
                prog_bar.progress(progress_status)
                        
        else:
            st.write('pass')
            pass
    
    elif process == 'Deployment':
        pass
    
    elif process == 'Monitoring':
        pass

else:
    st.write('hello')
