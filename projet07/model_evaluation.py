import matplotlib.pyplot as plt  # For plotting the ROC curve
import mlflow  # For MLflow logging and model management
import mlflow.sklearn  # Specifically for logging sklearn models in MLflow
from mlflow.tracking import MlflowClient  # For managing experiments and runs in MLflow
from mlflow.models.signature import infer_signature  # For inferring input-output signatures of models
from sklearn.metrics import roc_auc_score  # For evaluating model performance
from sklearn.metrics import roc_curve  # For calculating the ROC curve points
import numpy as np  # For handling numerical arrays and operations
import re  # For regular expressions to check run names
from typing import Tuple, Dict, Any  # For type hinting


def score_model(y_true: np.array,
                y_pred: np.array,
                cost_ratio:float = 10.0,
                verbose: bool = False):
    """
    Computes a custom score for credit approval models based on the predicted probabilities,
    considering the cost of false negatives (defaults) and the gains from true positives (approved clients).

    Parameters:
    -----------
    y_true : np.array
        The true binary labels (0 = no default, 1 = default).
    
    y_pred : np.array The predicted labels.
    
    cost_ratio : float, optional
        The ratio of the cost associated with a false negative (defaulting client)
        compared to the gain from a true positive (approved client) (default is 10.0).
    
    validation_threshold : float, optional
        The threshold for deciding whether to approve a client. Probabilities below this
        threshold result in approval (default is 0.1).
    
    verbose : bool, optional
        If True, prints detailed information about the calculations (default is False).

    Returns:
    --------
    total_score : float
        The calculated score based on the cost of defaults and the gains from approvals,
        normalized by the maximum possible loss and gain.

    Notes:
    ------
    - The function assumes that the cost of a defaulting client is `cost_ratio` times greater
      than the gain from an approved (non-defaulting) client.
    - Approved clients are determined based on the `validation_threshold`: clients with
      predicted default probabilities below the threshold are approved.
    """   

    max_loss = cost_ratio*sum(y_true)
    max_gain = 1.0*sum(1-y_true)

    losses = cost_ratio * sum( y_pred*y_true )
    gains =  1.0*sum( y_pred * (1 - y_true))

    total_score = (gains - losses + max_loss)/(max_gain+max_loss)
    if verbose:        
        print('approved:', y_pred, sep='\t')
        print('defaulted:', y_true, sep='\t')
        print('max_loss:', max_loss, sep='\t')
        print('max_gain:', max_gain, sep='\t')
        print('losses:', losses, sep='\t')
        print('gains:', gains, sep='\t')

    # Return the average score
    return total_score

def score_model_proba(y_true: np.array,
                y_pred_proba: np.array,
                cost_ratio:float = 10.0, 
                validation_threshold: float = 0.1, 
                verbose: bool = False) -> float:
    """
    Computes a custom score for credit approval models based on the predicted probabilities,
    considering the cost of false negatives (defaults) and the gains from true positives (approved clients).

    Parameters:
    -----------
    y_true : np.array
        The true binary labels (0 = no default, 1 = default).
    
    y_pred_proba : np.array
        The predicted probabilities of default from the model (between 0 and 1).
    
    cost_ratio : float, optional
        The ratio of the cost associated with a false negative (defaulting client)
        compared to the gain from a true positive (approved client) (default is 10.0).
    
    validation_threshold : float, optional
        The threshold for deciding whether to approve a client. Probabilities below this
        threshold result in approval (default is 0.1).
    
    verbose : bool, optional
        If True, prints detailed information about the calculations (default is False).

    Returns:
    --------
    total_score : float
        The calculated score based on the cost of defaults and the gains from approvals,
        normalized by the maximum possible loss and gain.

    Notes:
    ------
    - The function assumes that the cost of a defaulting client is `cost_ratio` times greater
      than the gain from an approved (non-defaulting) client.
    - Approved clients are determined based on the `validation_threshold`: clients with
      predicted default probabilities below the threshold are approved.
    """   
    # Apply the score_client function to the predicted probabilities
    y_pred = np.int64(y_pred_proba < validation_threshold)   
    total_score = score_model(y_true, y_pred, cost_ratio, verbose)

    # Return the average score
    return total_score

def get_trainned_model(model: Any, 
                       train_tupple: Tuple[np.array, np.array], 
                       model_name: str, 
                       model_version: int,
                       force_train: bool = False ) -> Any:
    """
    Loads a model from the MLflow Model Registry or trains it if necessary.

    Parameters:
    -----------
    model : Any
        The model object that will be trained if not found in the registry or
        if force training is enabled.
    
    train_tupple : Tuple[np.array, np.array]
        A tuple containing the training data:
        - train_tupple[0]: Features (X_train) for training.
        - train_tupple[1]: Target values (y_train) for training.

    model_name : str
        The name of the model as registered in the MLflow Model Registry.

    model_version : int
        The version of the model to load from the MLflow Model Registry.

    force_train : bool, optional
        If True, forces retraining of the model even if a version exists in
        the MLflow registry (default is False).

    Returns:
    --------
    model : Any
        The loaded or trained model object.

    Notes:
    ------
    - If the model exists in the MLflow Model Registry and `force_train` is False,
      the function will load the model from the registry.
    - If the model does not exist in the registry or if `force_train` is True, 
      the function will fit the model on the provided training data.
    """

    model_uri = f'models:/{model_name}/{model_version}'    
    
    try:
        model = mlflow.sklearn.load_model(model_uri=model_uri)
    except mlflow.exceptions.RestException:
        force_train= True
        
    if force_train:
        model.fit(train_tupple[0], train_tupple[1])

    return model

def record_model_run(model: Any,
                     test_tupple: Tuple[np.array, np.array],
                     experiment_name: str,
                     model_name: str,
                     model_version: int,
                     model_params: Dict[str, Any],
                     artifact_path: str,
                     validation_threshold: float = 0.1) -> Any:
    """
    Records a model run in MLflow by checking for duplicates and logging the model,
    parameters, and metrics if no duplicates are found.

    Parameters:
    -----------
    model : Any
        The trained model to be logged in MLflow.
    
    test_tupple : Tuple[np.array, np.array]
        A tuple containing the test data:
        - test_tupple[0]: Features (X_test) for testing.
        - test_tupple[1]: Target values (y_test) for testing.

    experiment_name : str
        The name of the experiment in MLflow.

    model_name : str
        The name of the model to be registered in MLflow.

    model_version : int
        The version of the model to check for in the MLflow Model Registry.

    model_params : Dict[str, Any]
        A dictionary of model parameters to log in MLflow.

    artifact_path : str
        The path where the model artifacts will be stored.

    validation_threshold : float, optional
        The threshold for model validation (default is 0.1).

    Returns:
    --------
    run : Any
        The MLflow run object corresponding to the logged model.
    """


    params = {}
    params = {key:str(val) for key, val in model_params.items()}
    params['validation_threshold'] = str(validation_threshold)
        
    # Estimate metrics
    y_pred = model.predict_proba(test_tupple[0])[:, 1]
    roc_auc = roc_auc_score(test_tupple[1].values, y_pred)
    model_score = score_model_proba(test_tupple[1].values, y_pred, validation_threshold=validation_threshold)
    metrics = {"roc_auc":roc_auc,
               "model_score":model_score}
    
    output ={'y_pred': y_pred,
             'roc_auc':roc_auc,
             'model_score': model_score,
             'validation_threshold':validation_threshold,
            }
    
    # Retrieve the runs for a specific experiment by name or ID
    client = MlflowClient()    
    mlflow.set_experiment(experiment_name)
    experiment = client.get_experiment_by_name(experiment_name)
    runs = []
    if experiment: runs = client.search_runs(experiment.experiment_id)
    
    # check registered runs 
    similar_runs = [-1]
    for run in runs:

        if (run.data.params == params) and (run.data.metrics == metrics) and (model_name == run.info.run_name.rsplit('_',1)[0]):
            print("Duplicate run found, not logging a new one.")
            output['run'] = run
            return  output
        
        similar_runs = similar_runs + [int(i) for i in re.findall(model_name+r'_\[(\d*)\]',run.info.run_name)]
    
    max_num = max(similar_runs)+1
    possible_nums = np.int64(np.linspace(0,max_num, max_num+1))
    mask = ~np.isin(possible_nums,similar_runs)
    run_num = min(possible_nums[mask])
    run_name = f'{model_name}_[{run_num:03d}]'
    
    print("Logging run...", end='\t')
    with mlflow.start_run(run_name=run_name) as run:
        # Train on the training data
        
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the metrics
        
        mlflow.log_metrics(metrics)
        
        # Log the model
    
        # Infer the model signature
        signature = infer_signature(test_tupple[0], y_pred, params)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=test_tupple[0][0:10,:],
            registered_model_name=model_name,
        )


    print('Done.')
    output['run'] = mlflow.get_run(model_info.run_id)
    return output


def plot_roc_curve(pred_test_pairs: Dict[str, Tuple[np.array, np.array]]) -> None:
    """
    Plots ROC curves for multiple models and displays the custom model score on the plot.

    Parameters:
    -----------
    pred_test_pairs : Dict[str, Tuple[np.array, np.array]]
        A dictionary where the key is the model name, and the value is a tuple:
        - tuple[0]: The true binary labels (y_test) for the test data.
        - tuple[1]: The predicted probabilities (y_pred) for the test data.
    

    Returns:
    --------
    None
        This function generates and shows a ROC curve plot with each model's curve 
        labeled by its name and custom score.
    """
    
    # Iterate over each model's predictions and true labels
    for model_name, values in pred_test_pairs.items():
        (y_tst, y_prd) = values[:2]
        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y_tst, y_prd)
        
        # Calculate the custom model score using score_model
        roc_auc = roc_auc_score(y_tst, y_prd)
        
        # Plot the ROC curve for the model
        plt.plot(fpr, tpr, label=f'{model_name}; roc_auc_score={roc_auc:0.4f}')
    
    # Customize the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    
    # Plot the reference line (random model)
    plt.plot([0, 1], [0, 1], ':k')
    
    # Display the plot
    plt.show()
