from projet07.model_evaluation import score_model, score_model_proba
import numpy as np


def test_score_model_size_1_full_FN()->None:
    """
    Test if score zero on false negative
    """
    y_true = np.array([True])
    y_pred = np.array([False])
    assert score_model(y_true=y_true, y_pred=y_pred) == 0

def test_score_model_size_1_full_FP()->None:
    """
    Test if score zero on false positive
    """
    y_true = np.array([0])
    y_pred = np.array([1])
    assert score_model(y_true=y_true, y_pred=y_pred) == 0

def test_score_model_proba_size_1_full_FN()->None:
    """
    Testing if it scores 0 when the estimated probability of defaulting is 
    below the validation threshold (credit approved) but client defaults
    """
    y_true = np.array([1])
    y_pred_proba = np.array([0.4])
    validation_threshold = 0.40001
    assert score_model_proba(y_true=y_true,
                             y_pred_proba=y_pred_proba,
                             validation_threshold=validation_threshold) == 0


def test_score_model_proba_size_1_full_FP()->None:
    """
    Testing if it scores 0 when the estimated probability of defaulting is 
    above the validation threshold (credit denied) but client doesn't default
    """
    y_true = np.array([0])
    y_pred_proba = np.array([0.4])
    validation_threshold = 0.4
    assert score_model_proba(y_true=y_true,
                             y_pred_proba=y_pred_proba,
                             validation_threshold=validation_threshold) == 0
    
def test_score_model_pair_fully_wrong():
    """"
    test score zero on absolute wrong predictions for a [False, True] array
    """

    y_true = [False, True]
    y_pred = [True, False]
    assert score_model(y_true, y_pred) == 0

def test_score_model_pair_fully_right():
    """"
    test score 1.0 on absolute right predictions for a [False, True] array
    """

    y_true = [False, True]
    y_pred = y_true
    assert score_model(y_true, y_pred) == 1    

def test_score_model_pair_half_right_but_costly():
    """
    test score (1)/(11) for having one true negative and one false negative with cost_ratio = 10.0
    """
    y_true = [False, True]
    y_pred = [False, False]
    cost_ratio = 10.0
    assert score_model(y_true, y_pred, cost_ratio=cost_ratio) == 1.0/11.0


def test_score_model_pair_half_right_but_acceptable():
    """
    test score (10)/(11) for having one False positive and one true negative with cost_ratio = 10.0
    """
    y_true = [False, True]
    y_pred = [True, True]
    cost_ratio = 10.0
    assert score_model(y_true, y_pred, cost_ratio=cost_ratio) == 1.0/11.0


