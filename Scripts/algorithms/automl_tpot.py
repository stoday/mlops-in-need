import threading
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import ctypes
import os

# 用物件執行 TPOT
class tpot_worker(threading.Thread):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0
        self.should_stop = threading.Event()
        self.dataset = dataset
        
    def run(self):
        X_dataset = self.dataset.iloc[:, 0:-1]
        y_dataset = self.dataset.iloc[:, -1]
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