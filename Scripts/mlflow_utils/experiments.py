
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
from pprint import pprint

def print_experiment_info(experiments):
    for exps in experiments:
        print("- experiment_id: {}, name: {}, lifecycle_stage: {}"
              .format(exps.experiment_id, exps.name, exps.lifecycle_stage))

def list_all(experiment_ids):
    client = MlflowClient()

    # # Delete the last experiment
    # client.delete_experiment(exp_id)

    # Fetch experiments by view type
    print("Active experiments:")
    print_experiment_info(client.list_experiments(view_type=ViewType.ACTIVE_ONLY))
    
    print('\n*** START ***\n')
    experiemnt_results = client.search_runs(experiment_ids=[experiment_ids], 
                                            order_by=['tag.start_time DESC'])
    pprint(experiemnt_results)
    print('\n*** END ***\n')
    
    exp_list_by_id = client.search_runs(experiment_ids=experiment_ids, order_by=['tag.start_time DESC'])
    exp_info_df = pd.DataFrame([])
    exp_data_df = pd.DataFrame([])
    exp_results = pd.DataFrame([])
    for x in exp_list_by_id:
        # experiment_info_dict = x.to_dictionary()['info']['start_time']
        # experiment_data_dict = x.to_dictionary()['data']['metrics']
        experiment_info_pd = pd.DataFrame(x.to_dictionary()['info'], index=[0])
        experiment_data_pd = pd.DataFrame(x.to_dictionary()['data']['metrics'], index=[0])
        exp_info_df = pd.concat([exp_info_df, experiment_info_pd], axis=0, ignore_index=True)
        exp_data_df = pd.concat([exp_data_df, experiment_data_pd], axis=0, ignore_index=True)
        exp_results = pd.concat([exp_info_df, exp_data_df], axis=1, ignore_index=True)
    
    return exp_results
    
    
