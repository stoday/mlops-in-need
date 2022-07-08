
import re
import os
import numpy as np

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
    # print(model_pipline)    # check
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
    # print(model_code)
    # exec(model_code)
    
    return model_code