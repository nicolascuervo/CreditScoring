import pandas as pd
import numpy as np
import optuna

def record_crossvalidation_stats(trial, model_scores, roc_aucs):
    # Log individual fold results
    for i, (ms, ra) in enumerate(zip(model_scores, roc_aucs)):
        trial.set_user_attr(f"trial_{i:2d}__roc_auc", ra)
        trial.set_user_attr(f"trial_{i:2d}__model_score", ms)

    # Log aggregated results    
    trial.set_user_attr(f"mean__roc_auc", np.mean(roc_aucs))
    trial.set_user_attr(f"std__roc_auc", np.std(roc_aucs))
    trial.set_user_attr(f"median__roc_auc", np.median(roc_aucs))
    
    trial.set_user_attr(f"mean__model_score", np.mean(model_scores))
    trial.set_user_attr(f"std__model_score", np.std(model_scores))        
    median_model_score =  np.median(model_scores)
    trial.set_user_attr(f"median__model_score", median_model_score )
    return median_model_score

def get_study_summary(study):
     trial = study.trials[0]
     values = [list(trial.params.values()) + trial.values + list(trial.user_attrs.values())]
     for i, trial in enumerate(study.trials[1:]):
          values.append(list(trial.params.values()) + trial.values + list(trial.user_attrs.values()))

     columns = list(study.best_params.keys())+['score']+list(trial.user_attrs.keys())     
     summary = pd.DataFrame(data=values, columns=columns)
     return summary

def print_study_evol(study):
    for i, trial in enumerate(study.trials):        
        if i == 0 : best_one = {"idx":i, 'value':trial.value}
        if trial.value >= best_one['value']:
            best_one = {"idx":i, 'value':trial.value}
        msg = f'[I {str(trial.datetime_complete)[:23]}] Trial {i} finished with value: {trial.value} and parameters: {trial.params}. '
        msg = msg + f'Best is trial {best_one['idx']} with value: {best_one['value']}.'
        print(msg)


<<<<<<< HEAD
def prune_incomplete_trials(study_name, database_url, max_trials=-1):

=======
def prune_incomplete_trials(study_name, database_url):
         
>>>>>>> b7241515be77b455c4ea907c2b5d7aae7a2eb95d
    study = optuna.create_study(study_name=study_name, 
                                direction="maximize", 
                                storage=database_url,                             
                                load_if_exists=True,)
    
<<<<<<< HEAD
    
=======
>>>>>>> b7241515be77b455c4ea907c2b5d7aae7a2eb95d
    new_study = optuna.create_study(study_name=study_name, 
                                direction="maximize", 
                                load_if_exists=True,
                                )
    
<<<<<<< HEAD
    for trial in study.trials[:max_trials]:
=======
    for trial in study.trials:
>>>>>>> b7241515be77b455c4ea907c2b5d7aae7a2eb95d
        if trial.value is not None:
            # trial.set_user_attr("history", 'test')
            new_study.add_trial(trial)
    n_pruned = len(study.trials) - len(new_study.trials) 

    
    
    if n_pruned>0:
        response  = input(f'{n_pruned} trials will be pruned, are you sure you want to continue? (Y/N)')
        if response.lower() != 'y':
            print('Prunning canceled')
            return study
    
    optuna.delete_study(study_name=study_name, storage=database_url)
    
    study = optuna.create_study(study_name=study_name, 
                                direction="maximize", 
                                storage=database_url,                             
                                load_if_exists=True,)
    
    for trial in new_study.trials:
        study.add_trial(trial)

    return study