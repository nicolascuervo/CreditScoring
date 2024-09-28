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
    best_one = None
    for i, trial in enumerate(study.trials):#19:32:53,389 19:32:51,982
        if (best_one is None) or (trial.value> best_one['value']):
            best_one = {"idx":i, 'value':trial.value}
        msg = f'[I {str(trial.datetime_complete)[:23]}] Trial {i} finished with value: {trial.value} and parameters: {trial.params}. '
        msg = msg + f'Best is trial {best_one['idx']} with value: {best_one['value']}.'
        print(msg)