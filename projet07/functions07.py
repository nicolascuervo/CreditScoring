

def score_client(p:float, cost_ratio:float=10.0):
    """
    Computes a custom client score based on the predicted probability 
    of default and a specified cost ratio for defaulting.

    Parameters:
    -----------
    p : float
        The predicted probability of default (must be between 0 and 1).
    cost_ratio : float, optional
        The cost ratio of a default compared to a full payment (default is 10.0).
        This value influences how steeply the score decreases as the probability 
        of default increases.

    Returns:
    --------
    float
        A score between 0 and 1 where:
        - 1 indicates no probability of default (p = 0),
        - 0 indicates a high probability of default (p = 1).
        The score decreases as the probability of default increases, with the
        steepness of the decline controlled by the cost_ratio parameter.
    """
    return (1.- p)/(1.+(cost_ratio-2.)*p)