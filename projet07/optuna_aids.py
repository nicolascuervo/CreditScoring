import pandas as pd
import numpy as np
import optuna
from multiprocessing import Pool
from typing import List, Tuple, Dict, Any, Union
import inspect
from imblearn.pipeline import Pipeline #type: ignore
from model_evaluation import score_model_proba
from sklearn.metrics import roc_auc_score

# Custom type annotation for suggest_instructions
SuggestInstruction = Dict[str, List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]]]
def evaluate_fold(args: Tuple) -> Tuple[float, float]:

    """ Function to evaluate a single fold in parallel. """
    X, y, model_steps, params, validation_threshold, train_idx, test_idx, rnd_stt = args

    # Intitialize a random state
    random_state = np.random.RandomState(rnd_stt)

    # Dynamically instantiate the steps for the pipeline
    steps = []
    for step_name, step_class, step_args, step_kwargs in model_steps:
        step_kwargs = step_kwargs.copy()
        # Update kwargs with random state if necessary
        if 'random_state' in inspect.signature(step_class.__init__).parameters:
            step_kwargs['random_state'] = random_state 
        #verify model_params that belong to this step
        param_kwargs = {k.split('__',1)[-1]: v for k, v in params.items() if k.startswith(step_name+'__')}
        step_kwargs.update(param_kwargs)
        step_instance = step_class(*step_args, **step_kwargs)
        steps.append((step_name, step_instance))

    # Create the pipeline with the dynamically generated steps
    pipeline = Pipeline(steps)

    # Fit the pipeline on the training data
    pipeline.fit(X[train_idx,:], y.iloc[train_idx])

    # Predict probabilities on the test data
    y_pred = pipeline.predict_proba(X[test_idx])[:, 1]

    # Evaluate model performance
    mdl_scr = score_model_proba(y.iloc[test_idx], y_pred, validation_threshold=validation_threshold, verbose=False)
    ra = roc_auc_score(y.iloc[test_idx], y_pred)

    return mdl_scr, ra

def create_objective(model_steps, suggest_instructions:SuggestInstruction, X, y, cv, 
                     processes:int=1,
                     random_states:List|int|None=None):
    
    def objective(trial:optuna.Trial)->float:
        # prepare random_states if any
        random_state_list = np.array([rs for rs in np.array([random_states]).reshape(-1)])
        if len(random_state_list)==1:
            random_state_list = np.repeat(random_state_list, len(cv))
        elif len(random_state_list)!=len(cv):
            raise ValueError(f"Array size {len(random_state_list)} does not match the expected size {len(cv)}.")
        
        # Step 1: Initialize parameters using Optuna
        params = suggest_params(trial, suggest_instructions)
            # pop validation threshold
        validation_threshold = params.pop('validation_threshold') 
        
        # prepare arguments for fold evaluation
        args_list = [(X, y, model_steps, params, validation_threshold, train_idx, test_idx, rnd_stt) 
                for rnd_stt, (train_idx, test_idx) in zip(random_state_list, cv)]
        
        if processes == 1: 
            results = [ evaluate_fold(args) for args in args_list ] 
        else:
            with Pool(processes=10) as pool:
                results = pool.map(evaluate_fold, args_list)

        # Collect the results from each fold
        model_scores, roc_aucs = zip(*results)
              
        # record fold scores
        median_model_score = record_crossvalidation_stats(trial, model_scores, roc_aucs)
        
        return median_model_score
    
    
    def suggest_params(trial: optuna.Trial, suggest_instructions: SuggestInstruction) -> dict:
        """
        Generate hyperparameters using Optuna's suggest functions based on the
        provided instructions.

        This function loops over a dictionary of instructions where each key
        corresponds to an Optuna suggest method ('int', 'float', 'categorical',
        'loguniform', etc.) and each value is a list of tuples. Each tuple 
        contains a parameter name, a tuple of positional arguments, and a 
        dictionary of keyword arguments to pass to the corresponding 
        `trial.suggest_*` method.

        Parameters
        ----------
        trial : optuna.Trial
            An Optuna Trial object used for hyperparameter optimization.
            
        suggest_instructions : Dict[str, List[Tuple[str, Tuple[Any, ...], 
                                Dict[str, Any]]]]
            A dictionary where keys represent the type of suggest method 
            ('int', 'float', 'categorical', etc.) and values are lists of 
            tuples. Each tuple contains:
            - str : The name of the hyperparameter.
            - Tuple[Any, ...] : Positional arguments for the suggest function.
            - Dict[str, Any] : Keyword arguments for the suggest function.
        
        Returns
        -------
        dict
            A dictionary of generated hyperparameters with parameter names as
            keys and the suggested values as values.

        Raises
        ------
        ValueError
            If an unknown suggestion type is provided in the 
            `suggest_instructions`.
        """
        params = {}
        
        # Loop over each type of suggestion in the dictionary
        for suggestion_type, instructions in suggest_instructions.items():
            
            for param_name, args, kwargs in instructions:
                # Conditional to call the appropriate suggest_* function
                name = param_name.rsplit('__')[-1] # TODO : Best to remove for log term development
                if suggestion_type == 'int':
                    params[param_name] = trial.suggest_int(name, *args, **kwargs)
                elif suggestion_type == 'float':
                    params[param_name] = trial.suggest_float(name, *args, **kwargs)
                elif suggestion_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(name, *args, **kwargs)
                elif suggestion_type == 'loguniform':
                    params[param_name] = trial.suggest_loguniform(name, *args, **kwargs)
                # Add more suggestion types as needed
                else:
                    raise ValueError(f"Unknown suggestion type: {suggestion_type}")
        
        return params

    return objective



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
        if trial.value is None:
            best_one = best_one            
        elif trial.value >= best_one['value']:
            best_one = {"idx":i, 'value':trial.value}
        msg = f'[I {str(trial.datetime_complete)[:23]}] Trial {i} finished with value: {trial.value} and parameters: {trial.params}. '
        msg = msg + f'Best is trial {best_one['idx']} with value: {best_one['value']}.'
        print(msg)


def prune_incomplete_trials(study_name, database_url, max_trials=-1):

    study = optuna.create_study(study_name=study_name, 
                                direction="maximize", 
                                storage=database_url,                             
                                load_if_exists=True,)
    
    
    new_study = optuna.create_study(study_name=study_name, 
                                direction="maximize", 
                                load_if_exists=True,
                                )
    
    for trial in study.trials[:max_trials]:
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