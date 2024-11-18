from imblearn.under_sampling.base import BaseUnderSampler
from sklearn.utils import check_X_y
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer

import numpy as np
import pandas as pd
import re

def clean_column_names(columns):
    """
    Clean column names by replacing problematic characters such as spaces,
    slashes, commas, etc. with underscores.

    Parameters:
    -----------
    columns : list of str
        The list of column names to clean.

    Returns:
    --------
    list of str
        The cleaned column names.
    """
    cleaned_columns = []
    for col in columns:
        # Replace spaces, slashes, commas, and other special characters with underscores
        clean_col = re.sub(r'[ /,]', '_', col)
        # Remove any remaining problematic characters
        clean_col = re.sub(r'[^a-zA-Z0-9_]', '', clean_col)
        cleaned_columns.append(clean_col)
    return cleaned_columns

class DataFrameWrapper(TransformerMixin):
    def __init__(self, transformer, feature_names=None, clean_column_names=True):
        self.transformer = transformer
        self.feature_names = feature_names
        self.feature_names_in_ = feature_names
        self.clean_column_names = clean_column_names

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        # If feature names are not provided, use the column names of X        
        if self.feature_names is None:
            self.feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_in_ = self.feature_names

        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:

        if (input_features is None) and (self.feature_names_in_ is None):
            return np.array([])
        elif (input_features is None):
            input_features = self.feature_names_in_
        
        return self.transformer.get_feature_names_out(input_features)


    def transform(self, X):
        transformed = self.transformer.transform(X)
        
        if hasattr(self.transformer, 'get_feature_names_out'):
            feature_names = self.transformer.get_feature_names_out(self.feature_names)
        else:
            feature_names = self.feature_names
        if self.clean_column_names:
            feature_names = clean_column_names(feature_names)
        return pd.DataFrame(transformed, columns=feature_names, index=X.index)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

class CreateDomainFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create domain-specific features.
    
    This transformer calculates the following new features:
    - CREDIT_INCOME_PERCENT: AMT_CREDIT / AMT_INCOME_TOTAL
    - ANNUITY_INCOME_PERCENT: AMT_ANNUITY / AMT_INCOME_TOTAL
    - CREDIT_TERM: AMT_ANNUITY / AMT_CREDIT
    - DAYS_EMPLOYED_PERCENT: DAYS_EMPLOYED / DAYS_BIRTH
    """
    composed_mapping = {'DAYS_EMPLOYED_ANOM': ['DAYS_EMPLOYED'],
                    'CREDIT_INCOME_PERCENT': ['AMT_CREDIT', 'AMT_INCOME_TOTAL'],
                    'ANNUITY_INCOME_PERCENT': ['AMT_ANNUITY', 'AMT_INCOME_TOTAL'],
                    'CREDIT_TERM': ['AMT_ANNUITY', 'AMT_CREDIT'],
                    'DAYS_EMPLOYED_PERCENT': ['DAYS_EMPLOYED', 'DAYS_BIRTH']}

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        

        return self  # No fitting necessary

    def transform(self, X):
        # Copy the input data to avoid modifying the original DataFrame
        X = X.copy()
        
        # Create the new domain-specific features
        X['DAYS_EMPLOYED_ANOM'] = X["DAYS_EMPLOYED"] == 365243
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace({365243: np.nan})
        X['DAYS_BIRTH'] = abs(X['DAYS_BIRTH'])
        X['CREDIT_INCOME_PERCENT'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['ANNUITY_INCOME_PERCENT'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['CREDIT_TERM'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['DAYS_EMPLOYED_PERCENT'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        
        self.feature_names_out_ = X.columns
        return X
 
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        new_features = ['DAYS_EMPLOYED_ANOM',         
         'CREDIT_INCOME_PERCENT',
         'ANNUITY_INCOME_PERCENT',
         'CREDIT_TERM',
         'DAYS_EMPLOYED_PERCENT']
        
        if input_features is None and self.feature_names_out_ is None:
            return new_features
        elif input_features is None :
            return self.feature_names_out_
                
        features_added = [feat for feat in new_features if feat not in input_features]
        feature_names_out = input_features.copy()
        feature_names_out = feature_names_out + features_added

        return feature_names_out
   
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

def get_encoding_transformers( X:pd.DataFrame, target_name:str='TARGET')->tuple[ColumnTransformer]:
    """
    Generates column transformers for encoding categorical and numerical features of a DataFrame.
    
    This function identifies categorical and numerical columns from the input DataFrame `X`, 
    determines which categorical columns are binary or non-binary,
    and returns two transformers:
    
    - A transformer for encoding categorical features as follows:
        - binary categorical features -> OrdinalEncoder ( 0 and 1)
        - other categorical features -> OneHotEncoder, nan values use another column
        - Non categorical features -> 'passthough' (except for target_name)
    - A transformer for dropping columns that end with '_nan'.

    Parameters:
    -----------
    X : pd.DataFrame
        The input DataFrame containing features for transformation.
        
    target_name : str, optional
        The name of the target column to exclude from transformations (default is 'TARGET').

    Returns:
    --------
    tuple[ColumnTransformer]:
        A tuple of two ColumnTransformers:
        - The first transformer applies encoding to categorical columns and leaves numerical columns as is.
        - The second transformer drops columns ending in '_nan'.

    Notes:
    ------
    - Categorical columns with two unique values (binary) are transformed using `OrdinalEncoder`.
    - Categorical columns with more than two unique values are transformed using `OneHotEncoder`.
    - Numerical columns are passed through without transformation.
    - Columns ending with '_nan' will be dropped by the second transformer.
    - The transformers are wrapped using a `DataFrameWrapper` to ensure output as a pandas DataFrame.
    """
    # identify categorical and numerical features
    columns = [col for col in X.columns if col!=target_name]
    cat_cols = [ col for col in columns if (X[col].dtype =='object')]
    num_cols = [columns[i] for i, mask in enumerate(~np.isin(columns, cat_cols,)) if mask]

    # check which categorical features are binary or not
    #   Use of len(X[col].unique()) instead of X[col].nunique() for nan values to be treated differently :
    #       Onehotencode will have t=2 false cols and a nan col will be droped later
    binary_cat_cols = [col for col in cat_cols if len(X[col].unique())<=2] 
    other_cat_cols = [cat_cols[i] for i, mask in enumerate(~np.isin(cat_cols, binary_cat_cols,)) if mask]

    # Create Columns transformers for each type of feature
    cat_enc_steps = [   ('binary_cat', OrdinalEncoder(), binary_cat_cols),  # For binary categorical columns
                        ('poly_cat', OneHotEncoder(), other_cat_cols),  # For other categorical columns
                        ('numerical','passthrough', num_cols),  # For numerical columns
                    ]

    encode_categorical = DataFrameWrapper(ColumnTransformer(cat_enc_steps, verbose_feature_names_out=False ))
    drop_nan_columns =  DataFrameWrapper(make_column_transformer(('passthrough', make_column_selector('^(?!.*_nan$).*')), 
                                                                verbose_feature_names_out=False))
    return encode_categorical, drop_nan_columns
                
    