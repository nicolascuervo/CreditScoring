import numpy as np  # For handling numerical arrays and operations

ArrayLikeBoolInt = list[bool|int] | np.ndarray[int|bool]

def score_model(y_true: ArrayLikeBoolInt,
                y_pred: ArrayLikeBoolInt,
                cost_ratio:float = 10.0,
                verbose: bool = False)->float:
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

    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    
    # Convert all `bool`, `int`, and `float` values to `int`
    y_true = np.rint(y_true).astype(int)
    y_pred = np.rint(y_pred).astype(int)

    max_loss = cost_ratio*sum(y_true)
    max_gain = 1.0*sum(1-y_true)
    
    losses = cost_ratio * sum( (1-y_pred)*y_true )
    gains =  1.0*sum((1-y_pred) * (1 - y_true))

    total_score = (gains - losses + max_loss)/(max_gain+max_loss)

    if verbose:        
        print('denied:', y_pred, sep='\t')
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
    y_pred = np.int64(y_pred_proba >= validation_threshold)   
    total_score = score_model(y_true, y_pred, cost_ratio, verbose)

    # Return the average score
    return total_score
    
