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
        self.session = []
          
    def run(self):
        X_dataset = self.dataset.iloc[:, 0:-1]
        y_dataset = self.dataset.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset)
        
        tpot = TPOTRegressor(generations=1, 
                             population_size=3, 
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
    