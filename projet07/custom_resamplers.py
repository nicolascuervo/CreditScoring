from imblearn.under_sampling.base import BaseUnderSampler
from sklearn.utils import check_X_y
from sklearn.model_selection import train_test_split

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
