from imblearn.under_sampling.base import BaseUnderSampler
from sklearn.utils import check_X_y
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import numpy as np

class CreateDomainFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create domain-specific features.
    
    This transformer calculates the following new features:
    - CREDIT_INCOME_PERCENT: AMT_CREDIT / AMT_INCOME_TOTAL
    - ANNUITY_INCOME_PERCENT: AMT_ANNUITY / AMT_INCOME_TOTAL
    - CREDIT_TERM: AMT_ANNUITY / AMT_CREDIT
    - DAYS_EMPLOYED_PERCENT: DAYS_EMPLOYED / DAYS_BIRTH
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self  # No fitting necessary

    def transform(self, X):
        # Copy the input data to avoid modifying the original DataFrame
        X = X.copy()
        
        # Create the new domain-specific features
        X['CREDIT_INCOME_PERCENT'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['ANNUITY_INCOME_PERCENT'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['CREDIT_TERM'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['DAYS_EMPLOYED_PERCENT'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        self.feature_names_out_ = X.columns
        return X
    
class AlwaysZeroClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that always predicts class 0.
    """
    def fit(self, X, y=None):
        # No fitting process needed as it always predicts 0
        return self

    def predict(self, X):
        # Always predict 0 for any input
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        # Return probabilities: 100% confidence for class 0
        return np.column_stack([np.ones(X.shape[0]), np.zeros(X.shape[0])])


class StratifiedUnderSampler(BaseUnderSampler):
    def __init__(self, sample_size: int = 1000, random_state = None, **kwargs):
        """
        StratifiedSampler using imbalanced-learn's BaseUnderSampler as a base class.

        Parameters
        ----------
        sample_size : int, optional (default=1000)
            The number of samples to draw during stratified sampling.
        
        random_state : int, optional (default=None)
            Seed for random number generator for reproducibility.
        
        kwargs : additional arguments passed to train_test_split
            Any additional arguments to be passed to the train_test_split function.
        """
        super().__init__(sampling_strategy='auto')
        self.sample_size = sample_size
        self.random_state = random_state
        self.kwargs = kwargs

    def _fit_resample(self, X, y):
        """
        Resample the dataset using stratified sampling during the fitting stage.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        
        y : array-like
            Target values.
        
        Returns
        -------
        X_resampled : array-like
            Resampled feature matrix.
        
        y_resampled : array-like
            Resampled target vector.
        """
        X, y = check_X_y(X, y)
        
        # Perform stratified sampling using train_test_split
        _, X_resampled, _, y_resampled = train_test_split(
            X, y, test_size=self.sample_size / len(X),
            stratify=y, random_state=self.random_state,
            **self.kwargs
        )
        
        return X_resampled, y_resampled
    def resample(self, X, y):
        """
        Public method to perform stratified resampling.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        
        y : array-like
            Target values.

        Returns
        -------
        X_resampled : array-like
            Resampled feature matrix.
        
        y_resampled : array-like
            Resampled target vector.
        """
        return self._fit_resample(X, y)
    
    def _more_tags(self):
        """
        Helper function for scikit-learn compatibility, indicating that the estimator requires `y`.

        Returns
        -------
        dict
            A dictionary of scikit-learn tags.
        """
        return {'requires_y': True}
